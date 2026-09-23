// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AgentsAPI } from '@e2e-tests/api/agents';
import {
  generateShortTestResourceName,
  MOCKS_DIR,
  PROJECT_ROOT,
  VERY_LONG_OPERATION_TIMEOUT,
} from '@e2e-tests/utils/constants';
import {
  disableAuthForTest,
  expectToastIsVisible,
  waitForLongOperation,
} from '@e2e-tests/utils/pageUtils';
import { expect, test as baseTest } from '@playwright/test';
import { parseDataFile } from '@studio/components/FileRowEditor/parse';
import fs from 'fs/promises';
import path from 'path';

const WORKSPACE = 'default';

/** The Fabric (`nemo-agents-spec-v1`) Email Security Triage example, uploaded through the
 *  New Agent flow exactly as a user would point the directory picker at it. Lives outside
 *  this package, so it is resolved from the repo root rather than from `MOCKS_DIR`. */
const REPO_ROOT = path.resolve(PROJECT_ROOT, '../../..');
const AGENT_DIR = path.join(
  REPO_ROOT,
  'plugins/nemo-agents/examples/nemo-agent-config/email-security-triage'
);

/** The small labelled email set the run is scored over. The UI's dataset upload takes JSON or
 *  JSONL — `inspectDatasetFile` rejects anything else — so the CSV is converted below rather
 *  than handed over as-is. */
const DATASET_CSV = path.join(
  PROJECT_ROOT,
  'public/sample-agents/email-phishing-analyzer/smaller_test.csv'
);
const EVAL_CONFIG = path.join(MOCKS_DIR, 'email-security-triage', 'eval-config.yaml');

/** Columns the eval config's `prompt_template` and metric read. The CSV carries more
 *  (`intents`, `source`, ...); they would reach the agent as noise, so they are dropped. */
const DATASET_COLUMNS = ['subject', 'sender', 'body', 'label'] as const;

/** The CSV as JSONL, one object per row, with the row count the run should report back. */
const datasetAsJsonl = async (): Promise<{ buffer: Buffer; rowCount: number }> => {
  const rows = parseDataFile(await fs.readFile(DATASET_CSV, 'utf8'), 'csv');
  const records = rows.map((row) =>
    Object.fromEntries(DATASET_COLUMNS.map((column) => [column, row[column] ?? '']))
  );
  expect(records.length, 'the sample CSV should hold rows to score').toBeGreaterThan(0);
  return {
    buffer: Buffer.from(records.map((record) => JSON.stringify(record)).join('\n'), 'utf8'),
    rowCount: records.length,
  };
};

/** The job name carried by a submit response, or undefined when the payload is not the shape
 *  this flow depends on: an object with a non-empty string `name`. */
const jobNameOf = (payload: unknown): string | undefined => {
  if (typeof payload !== 'object' || payload === null) return undefined;
  const name = (payload as { name?: unknown }).name;
  return typeof name === 'string' && name.length > 0 ? name : undefined;
};

interface AgentTracker {
  /** Register an agent for teardown: its deployments, the agent, and its spec fileset. */
  trackAgent: (name: string) => void;
  /** Register an experiment for teardown: the experiment and the fileset holding its run. */
  trackExperiment: (name: string) => void;
}

interface TestFixtures {
  agentsAPI: AgentsAPI;
  tracked: AgentTracker;
}

const test = baseTest.extend<TestFixtures>({
  agentsAPI: async ({ request }, runFixture) => {
    await runFixture(new AgentsAPI(request));
  },
  // Teardown runs even when the test fails, so anything registered here is cleaned up no
  // matter where in the UI flow we threw. A cleanup failure is recorded as an annotation
  // rather than thrown: it must not overwrite the test's own verdict, but it must not vanish
  // either — a swallowed one leaks the agent, its deployment and its filesets with no trace.
  // The API layer already tolerates a 404, so anything reaching here is a real failure.
  tracked: async ({ agentsAPI }, runFixture, testInfo) => {
    const agents: string[] = [];
    const experiments: string[] = [];

    await runFixture({
      trackAgent: (name) => agents.push(name),
      trackExperiment: (name) => experiments.push(name),
    });

    const cleanUp = async (what: string, remove: () => Promise<void>) => {
      try {
        await remove();
      } catch (error) {
        testInfo.annotations.push({
          type: 'cleanup-failed',
          description: `${what}: ${error instanceof Error ? error.message : String(error)}`,
        });
      }
    };

    for (const experiment of experiments) {
      await cleanUp(`fileset ${experiment}-data`, () =>
        agentsAPI.deleteFileset(WORKSPACE, `${experiment}-data`)
      );
      await cleanUp(`experiment ${experiment}`, () =>
        agentsAPI.deleteExperiment(WORKSPACE, experiment)
      );
    }
    for (const agent of agents) {
      await cleanUp(`deployments of ${agent}`, () => agentsAPI.removeDeployments(WORKSPACE, agent));
      await cleanUp(`agent ${agent}`, () => agentsAPI.deleteAgent(WORKSPACE, agent));
      // Deleting an agent leaves its spec fileset behind by design.
      await cleanUp(`fileset ${agent}-spec`, () =>
        agentsAPI.deleteFileset(WORKSPACE, `${agent}-spec`)
      );
    }
  },
});

test.describe('Agent Evaluation', () => {
  test.beforeEach(() => {
    test.skip(
      !process.env.RECORD,
      'Needs a live platform with an inference provider key to deploy and score an agent; run via pnpm test:e2e:record'
    );
  });
  test.beforeEach(async ({ page }) => disableAuthForTest(page));

  test('Deploys the Fabric email security triage agent and scores it on the sample CSV @record', async ({
    page,
    tracked,
  }) => {
    // Deploying a Fabric agent and scoring every row of the dataset against it are both
    // model-bound, so the whole flow gets one generous budget rather than the 1-minute default.
    test.setTimeout(30 * 60 * 1000);

    const suffix = generateShortTestResourceName().replace(/_/g, '-');
    // Agent names are lowercase letters, numbers and hyphens; the experiment name and the
    // `<experiment>-data` fileset derived from it follow the same entity-naming rules.
    const agentName = `email-security-triage-${suffix}`;
    const experimentName = `email-triage-${suffix}`;
    const evaluationName = `email-triage-run-${suffix}`;

    // Registered before the UI is touched, so a failure anywhere below still cleans up.
    tracked.trackAgent(agentName);
    tracked.trackExperiment(experimentName);

    await test.step('Upload the Fabric agent directory', async () => {
      await page.goto(`workspaces/${WORKSPACE}/agents`);
      await waitForLongOperation(page);

      await page.getByRole('button', { name: 'New Agent' }).click();
      const modal = page.getByRole('dialog', { name: 'Instrument an agent with NeMo Helix' });
      await expect(modal).toBeVisible();

      await modal.getByRole('tab', { name: 'Upload agent' }).click();
      // A webkitdirectory input takes a directory path; the browser hands over every
      // descendant, and the modal filters __pycache__ and friends out client-side.
      await modal.getByTestId('agent-directory-input').setInputFiles(AGENT_DIR);
      await expect(modal.getByText(/email-security-triage — \d+ files/)).toBeVisible();

      // The picker seeds a name from the config; this run needs a unique one.
      await modal.getByRole('textbox', { name: 'Name' }).fill(agentName);
      await modal.getByRole('button', { name: 'Create', exact: true }).click();
    });

    await test.step('Land on the new agent with nothing deployed', async () => {
      await expectToastIsVisible(page, `Agent "${agentName}" created`);
      await page.waitForURL(new RegExp(`/agents/${agentName}`));
      await expect(page.getByText('No deployments')).toBeVisible();
    });

    await test.step('Deploy the agent as a subprocess', async () => {
      await page.getByRole('button', { name: 'Deploy', exact: true }).click();

      const modal = page.getByRole('dialog', { name: 'Deploy Agent' });
      await expect(modal).toBeVisible();
      // Subprocess is the default runtime and the only one that needs no image.
      await modal.getByRole('button', { name: 'Deploy', exact: true }).click();

      await expectToastIsVisible(page, 'Deployment started successfully');
      await expect(modal).toBeHidden();
    });

    await test.step('Wait for the deployment to come up healthy', async () => {
      // The header pill reads Healthy only once a deployment of this agent reports `running`,
      // which is the gate the evaluation depends on: a run submitted against an agent that is
      // still starting fails on the first request.
      await expect(page.getByText('Healthy')).toBeVisible({
        timeout: VERY_LONG_OPERATION_TIMEOUT,
      });

      // The tab lists deployments as flex entries rather than table rows, so this reads the
      // entry's own name and status badge instead of a row.
      await page.getByRole('tab', { name: 'Deployments' }).click();
      const deployments = page.getByRole('tabpanel', { name: 'Deployments' });
      await expect(deployments.getByText(new RegExp(`^${agentName}-`))).toBeVisible();
      await expect(deployments.getByText('Running')).toBeVisible();
    });

    let jobName = '';
    const dataset = await datasetAsJsonl();

    await test.step('Submit an evaluation over the sample CSV', async () => {
      await page.getByRole('button', { name: 'Run evaluation' }).click();

      const modal = page.getByRole('dialog', { name: 'Run Agent Evaluation' });
      await expect(modal).toBeVisible();

      await modal.getByRole('textbox', { name: 'Experiment Name' }).fill(experimentName);
      await modal.getByRole('textbox', { name: 'Evaluation Name' }).fill(evaluationName);

      // Both pickers render the same upload input, and each unmounts its own once a file is
      // picked. Taking the config (the second) first leaves exactly one input behind, so
      // neither pick depends on an index that shifts underneath it.
      const fileInputs = modal.getByTestId('nv-upload-input-element');
      await expect(fileInputs).toHaveCount(2);
      await fileInputs.nth(1).setInputFiles(EVAL_CONFIG);
      await expect(fileInputs).toHaveCount(1);
      await fileInputs.first().setInputFiles({
        name: 'smaller_test.jsonl',
        mimeType: 'application/x-ndjson',
        buffer: dataset.buffer,
      });
      await expect(fileInputs).toHaveCount(0);

      // The job name is only reported in the submit response; the results page is addressed
      // by it, and the row assertion below needs it too.
      const submitted = page.waitForResponse(
        (response) =>
          response.request().method() === 'POST' &&
          /\/evaluator\/.*\/evaluate\/jobs$/.test(new URL(response.url()).pathname)
      );
      await modal.getByRole('button', { name: 'Submit' }).click();

      // Read as `unknown` and narrowed: asserting the shape would let a null payload throw a
      // bare TypeError here, and a non-string `name` through to fail later as a puzzling 404
      // on the results URL.
      const payload: unknown = await (await submitted).json();
      const submittedName = jobNameOf(payload);
      if (submittedName === undefined) {
        throw new Error(
          `The submit response did not name the job it created: ${JSON.stringify(payload)}`
        );
      }
      jobName = submittedName;

      await expect(modal).toBeHidden({ timeout: VERY_LONG_OPERATION_TIMEOUT });
    });

    await test.step('See the run listed against the agent', async () => {
      await page.waitForURL(/tab=evaluations/);
      await expect(page.getByRole('row', { name: new RegExp(jobName) })).toBeVisible({
        timeout: VERY_LONG_OPERATION_TIMEOUT,
      });
    });

    await test.step('Read the score off the finished run', async () => {
      // The results route polls the job every 5s and swaps the placeholder for the scores
      // table once it reaches a terminal state, so the wait happens on screen.
      await page.goto(`workspaces/${WORKSPACE}/evaluation/results/${jobName}`);
      await expect(page.getByText(jobName).first()).toBeVisible();

      // The metric cell carries the config's metric name; the row-results table below has no
      // column that repeats it, so this reaches the aggregate score and nothing else.
      const scoreRow = page.getByRole('row').filter({ hasText: 'string-check' }).first();
      await expect(scoreRow).toBeVisible({ timeout: 20 * 60 * 1000 });
      // formatScore renders a finite mean as a 3-decimal number and anything else as a dash,
      // so this asserts the run produced a real score, not that it scored well.
      await expect(scoreRow).toContainText(/\d\.\d{3}/);
    });

    await test.step('Show what the agent actually answered, row by row', async () => {
      // The score alone cannot distinguish a real run from one that scored nothing, so open
      // the per-row results: they carry the agent's own output text for every row.
      await page.getByRole('button', { name: `Row Results (${dataset.rowCount})` }).click();

      // A response cell holds a verdict *and* the reasoning the model wrote after it; the
      // Expected column holds the bare label alone. Requiring a word after the verdict is
      // what separates generated text from a ground-truth badge.
      const responses = page.getByRole('cell').filter({ hasText: /^(phishing|benign)\s+\w+/ });
      await expect(responses.first()).toBeVisible();
    });

    await test.step('Open the run logs', async () => {
      await page.getByRole('button', { name: 'Logs' }).click();
      // The evaluate job's own status logs. They record the job's steps, not the inference —
      // the deployment log below is where the model calls are visible.
      await expect(page.getByRole('button', { name: 'Wrap lines' })).toBeVisible();
    });

    await test.step('Show the agent serving one request per row', async () => {
      // The agent's deployment log is the request-side evidence: the Fabric server records an
      // access line per chat-completion it served, so the run leaves one per dataset row.
      await page.goto(`workspaces/${WORKSPACE}/agents/${agentName}?tab=logs`);
      const logs = page.getByRole('tabpanel', { name: 'Logs' });
      await expect(logs.getByText(/POST \/v1\/chat\/completions/).first()).toBeVisible({
        timeout: VERY_LONG_OPERATION_TIMEOUT,
      });

      // Counted from the panel's text rather than by locator: the viewer renders the log as
      // one block, so a per-line locator would match its container as well as its lines.
      const served = (
        (await logs.innerText()).match(/POST \/v1\/chat\/completions HTTP\/1\.1" 200/g) ?? []
      ).length;
      expect(served, 'the agent should have served one scored request per dataset row').toBe(
        dataset.rowCount
      );
    });
  });
});
