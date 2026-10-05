// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NHX_BASE_URL } from '@e2e-tests/utils/environment';
import { APIRequestContext, APIResponse } from '@playwright/test';

interface AgentDeployment {
  name?: string;
  agent?: string;
  status?: string;
}

/** Raise unless the response succeeded, or reported the resource already gone.
 *
 *  Only 404 is tolerated. Treating any other failure as success is what makes a leak silent:
 *  a 500 from the list call reads as "no deployments left", and a 409 from a delete reads as
 *  "deleted" — both leave the resource on the platform with nothing to show for it. */
const assertDone = (response: APIResponse, what: string, tolerateMissing = true): void => {
  if (response.ok() || (tolerateMissing && response.status() === 404)) return;
  throw new Error(`Failed to ${what}: ${response.status()} ${response.statusText()}`);
};

/** Direct API access to the entities an agent run leaves behind, for fixture teardown.
 *
 *  Deleting an agent is rejected while one of its deployments is still active, so a caller
 *  tears the deployments down first and waits for them to disappear. */
export class AgentsAPI {
  constructor(private request: APIRequestContext) {}

  private workspacePath(workspace: string) {
    return `${NHX_BASE_URL}/apis/agents/v2/workspaces/${encodeURIComponent(workspace)}`;
  }

  async listDeploymentsForAgent(workspace: string, agent: string): Promise<AgentDeployment[]> {
    const response = await this.request.get(`${this.workspacePath(workspace)}/deployments`);
    // No 404 to tolerate: the collection endpoint exists whether or not it holds anything.
    assertDone(response, `list deployments in ${workspace}`, false);
    const body = (await response.json()) as { data?: AgentDeployment[] };
    return (body.data ?? []).filter((deployment) => deployment.agent === agent);
  }

  async deleteDeployment(workspace: string, name: string) {
    const response = await this.request.delete(
      `${this.workspacePath(workspace)}/deployments/${encodeURIComponent(name)}`
    );
    assertDone(response, `delete deployment ${name}`);
  }

  async deleteAgent(workspace: string, name: string) {
    const response = await this.request.delete(
      `${this.workspacePath(workspace)}/agents/${encodeURIComponent(name)}`
    );
    assertDone(response, `delete agent ${name}`);
  }

  async deleteFileset(workspace: string, name: string) {
    const response = await this.request.delete(
      `${NHX_BASE_URL}/apis/files/v2/workspaces/${encodeURIComponent(workspace)}/filesets/${encodeURIComponent(name)}`
    );
    assertDone(response, `delete fileset ${name}`);
  }

  /** Deleting an experiment soft-deletes the evaluations whose only membership was that group,
   *  so a run's published results need no separate cleanup call. */
  async deleteExperiment(workspace: string, name: string) {
    const response = await this.request.delete(
      `${NHX_BASE_URL}/apis/intake/v2/workspaces/${encodeURIComponent(workspace)}/experiments/${encodeURIComponent(name)}`
    );
    assertDone(response, `delete experiment ${name}`);
  }

  /** Delete every deployment of an agent and wait for the controller to drop them, so the
   *  agent delete that follows is not rejected as still-deployed. */
  async removeDeployments(workspace: string, agent: string, timeout = 2 * 60 * 1000) {
    for (const deployment of await this.listDeploymentsForAgent(workspace, agent)) {
      if (deployment.name) await this.deleteDeployment(workspace, deployment.name);
    }

    const deadline = Date.now() + timeout;
    for (;;) {
      const remaining = await this.listDeploymentsForAgent(workspace, agent);
      if (remaining.length === 0) return;
      if (Date.now() >= deadline) {
        const names = remaining.map((deployment) => deployment.name ?? '<unnamed>').join(', ');
        throw new Error(
          `Deployments of ${agent} were still present after ${timeout}ms: ${names}. The agent cannot be deleted until they are gone.`
        );
      }
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }
  }
}
