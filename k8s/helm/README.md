<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Platform Helm Chart

![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square)

Documentation can be found at: https://docs.nvidia.com/nemo-platform.
For deployment instructions, see https://docs.nvidia.com/nemo-platform/documentation/kubernetes-deployment/setup.

## Platform Secrets Encryption Key

The platform secrets service reads `NMP_SECRETS_DEFAULT_ENCRYPTION_KEY` from the
API env Secret. The value must be base64-encoded and decode to at least 32 bytes.

Set `secrets.defaultEncryptionKey.value` to provide your own key. When that value
is empty and neither `envFromSecret` nor
`secrets.defaultEncryptionKey.existingSecret.name` is set, the chart runs a
pre-install hook that creates `<fullname>-api-env` with a per-install random key.
The hook is install-only and refuses to patch or rotate an existing Secret.

Set `secrets.defaultEncryptionKey.existingSecret.name` to use an existing Secret
for only the secrets service encryption key. After Kubernetes decodes the Secret
data, the value loaded from `secrets.defaultEncryptionKey.existingSecret.key`
must be a base64-encoded key that decodes to at least 32 bytes.

Set `envFromSecret` to use a fully user-managed API env Secret. In that mode the
chart does not create or generate the API env Secret.

On upgrade, the generated Secret must already exist and contain
`NMP_SECRETS_DEFAULT_ENCRYPTION_KEY`. If it is missing, restore the original
Secret instead of generating a replacement key; existing encrypted platform
secrets will not decrypt with a new key.

## Intake and ClickHouse

Intake is included in the platform API service group. By default, the chart
deploys a single-node embedded ClickHouse 26.3 LTS service, and Intake creates
and migrates its own `intake` database on first use.

The embedded ClickHouse is intended for development, evaluation, and
non-critical single-node installations. It does not provide replication or
automatic backups. For a production deployment that requires high availability,
set `clickhouse.enabled` to `false` and provide a separately managed ClickHouse:

First, create the credentials Secret in the Helm release namespace. The Secret
key must match `externalClickhouse.existingSecretPasswordKey`:

```shell
kubectl create secret generic clickhouse-credentials \
  --namespace <release-namespace> \
  --from-literal=password='<clickhouse-password>'
```

Then configure the external connection:

```yaml
clickhouse:
  enabled: false

externalClickhouse:
  host: clickhouse.example.internal
  port: 8443
  secure: true
  user: nemo
  database: intake
  existingSecret: clickhouse-credentials
  existingSecretPasswordKey: password
```

The external user must be allowed to create the configured database, tables,
materialized views, and indexes because Intake owns its ClickHouse migrations.

### ClickHouse sizing

Use retained span count as an initial operational threshold, not as a disk-size
estimate. Span `input`, `output`, and attribute payloads vary substantially.
Measure `bytes_on_disk` from representative traffic before setting production
storage:

```sql
SELECT
    table,
    sum(rows) AS physical_rows,
    formatReadableSize(sum(bytes_on_disk)) AS disk
FROM system.parts
WHERE active AND database = 'intake'
GROUP BY table
ORDER BY table;
```

The following are starting points for the current Intake schema and interactive
query workload:

| Retained Intake spans | Topology | ClickHouse resources | Storage |
| --- | --- | --- | --- |
| Up to 1 million | Embedded single node for non-critical workloads | 2 vCPU, 8 GiB RAM | Fast SSD, at least 20 GiB and 2× measured active data |
| 1–10 million | External preferred; embedded only when downtime and data loss are acceptable | 4–8 vCPU, 16–32 GiB RAM | Provisioned-IOPS SSD, at least 100 GiB and 2× measured active data |
| More than 10 million, or any HA requirement | Managed ClickHouse or an operator-managed replicated cluster | Start at 8 vCPU and 32 GiB RAM per replica, then load-test the actual ingest/read mix | Size from measured compression, retention, replication, and merge headroom |

Intake currently retains spans and its trace index for 90 days. Evaluator
results and annotations do not have a time-based TTL, so include their continuing
growth in capacity planning. ReplacingMergeTree retries also leave physical row
versions until background merges complete.

For large or frequently queried deployments, ClickHouse recommends:

- At least 8 GiB RAM even at low data volumes.
- A general-purpose starting ratio of 4 GiB RAM per CPU core.
- Provisioned-IOPS SSDs for latency-sensitive workloads.
- Roughly 1:30 to 1:50 RAM-to-storage for frequently accessed large datasets.
- Replication for production durability; vertically scale replicas before
  adding shards.

Validate CPU, query peak memory, active parts, merge backlog, disk latency, and
free space under representative batched OTLP ingestion before promoting a tier.
See the upstream
[ClickHouse sizing guide](https://clickhouse.com/docs/guides/oss/best-practices/sizing-and-hardware-recommendations)
and
[OSS operational recommendations](https://clickhouse.com/docs/guides/oss/best-practices/tips).

## Kyverno

The chart does not install Kyverno. Multi-node NCCL device injection renders
ClusterPolicies that Kyverno must apply. Enable exactly one cloud provider
under `multinodeNetworking`.
How-to: https://docs.nvidia.com/nemo-platform/latest/documentation/kubernetes-deployment/setup/helm/multinode-networking

## Volcano

The chart does not install Volcano. Multi-node `volcano_job` workloads need it.
`rbac.volcanoEnabled` defaults to true so the core controller can manage Volcano
CRs. Skip Volcano and set `rbac.volcanoEnabled: false` if you are not running
those jobs.
How-to: https://docs.nvidia.com/nemo-platform/latest/documentation/kubernetes-deployment/setup/helm/volcano

## OpenSandbox

The chart does not install OpenSandbox. Sandboxed GRPO / NeMo Gym talks to an
already installed server as an HTTP client (`OPEN_SANDBOX_DOMAIN`,
`OPEN_SANDBOX_API_KEY`). Gym sample YAML uses `OPENSANDBOX_*`; do not mix names.

`[kubernetes] namespace` in the server TOML **must be the Helm release
namespace**. Control plane may stay in `opensandbox-system`. Copy the API-key
Secret into the job namespace.

Example overlays: [k8s/helm/examples/opensandbox](https://github.com/NVIDIA-NeMo/nemo-platform/tree/main/k8s/helm/examples/opensandbox).
Shared-kernel (cluster default OCI runtime): https://docs.nvidia.com/nemo-platform/latest/documentation/kubernetes-deployment/setup/helm/open-sandbox
Kata QEMU: https://docs.nvidia.com/nemo-platform/latest/documentation/kubernetes-deployment/setup/helm/opensandbox-kata

## NetworkPolicies

The chart can render NetworkPolicy resources for the Platform API, core
controller, and managed job pods. A top-level switch enables all default
policies, and each subpolicy can be disabled independently for cluster-specific
exceptions.

## Values

For the complete default values, see [values.yaml](values.yaml).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| api | object | [See values.yaml](values.yaml#L765) | API configuration settings for the api deployment |
| api.affinity | object | `{}` | Affinity configuration for the API service. |
| api.annotations | object | `{}` | Annotations to add to the API service deployment. |
| api.autoscaling | object | [See values.yaml](values.yaml#L894) | Specifies autoscaling configurations for the deployment. |
| api.autoscaling.annotations | object | `{}` | Annotations for the HorizontalPodAutoscaler. |
| api.autoscaling.enabled | bool | `false` | Whether to enable horizontal pod autoscaler. |
| api.autoscaling.maxReplicas | int | `10` | The maximum number of replicas for the deployment. |
| api.autoscaling.minReplicas | int | `1` | The minimum number of replicas for the deployment. |
| api.autoscaling.targetCPUUtilizationPercentage | int | `80` | The target CPU utilization percentage. Requires api.resources.requests.cpu. |
| api.autoscaling.targetMemoryUtilizationPercentage | string | `nil` | The target memory utilization percentage. Requires api.resources.requests.memory. |
| api.enabled | bool | `true` | Specifies whether to enable the api deployment. |
| api.extraArgs | list | `[]` | Additional arguments to pass to the Platform API service |
| api.extraVolumeMounts | list | `[]` | Additional volume mounts to add to the Platform API container. |
| api.extraVolumes | list | `[]` | Additional volumes to add to the Platform API pod. |
| api.image | object | [See values.yaml](values.yaml#L771) | Container image configuration for the api deployment. |
| api.image.pullPolicy | string | `"IfNotPresent"` | The image pull policy determining when to pull new images. |
| api.image.repository | string | `"nvcr.io/nvidia/nemo-platform/nmp-api"` | The registry where the NeMo Platform image is located. |
| api.image.tag | string | `""` | The image tag to use. |
| api.livenessProbe | object | [See values.yaml](values.yaml#L853) | Liveness probe configuration for the api service. |
| api.livenessProbe.failureThreshold | int | `3` | The failure threshold for the liveness probe. |
| api.livenessProbe.httpGet | object | `{"path":"/health/live","port":"http"}` | The HTTP GET request to use for the liveness probe. |
| api.livenessProbe.periodSeconds | int | `10` | The frequency in seconds to perform the liveness probe. |
| api.livenessProbe.timeoutSeconds | int | `5` | The timeout in seconds for the liveness probe. |
| api.nodeSelector | object | `{}` | Node selector configuration for the API service. |
| api.podAnnotations | object | `{}` | Annotations to add to the API service pod. |
| api.podDisruptionBudget | object | [See values.yaml](values.yaml#L881) | PodDisruptionBudget configuration for the API service. |
| api.podDisruptionBudget.annotations | object | `{}` | Annotations for the PodDisruptionBudget. |
| api.podDisruptionBudget.enabled | bool | `false` | Whether to create a PodDisruptionBudget for the API pods. |
| api.podDisruptionBudget.minAvailable | int | `1` | Minimum number of API pods that must remain available during voluntary disruptions. Only one of minAvailable or maxUnavailable may be set. |
| api.podLabels | object | `{}` | Labels for the API service pod. |
| api.podSecurityContext | object | [See values.yaml](values.yaml#L818) | Pod-level security context settings for the API service. |
| api.podSecurityContext.fsGroup | int | `1000` | The file system group ID to use for all containers. |
| api.readinessProbe | object | [See values.yaml](values.yaml#L867) | Readiness probe configuration for the api service. |
| api.readinessProbe.failureThreshold | int | `3` | The failure threshold for the readiness probe. |
| api.readinessProbe.httpGet | object | `{"path":"/health/ready","port":"http"}` | The HTTP GET request to use for the readiness probe. |
| api.readinessProbe.periodSeconds | int | `10` | The frequency in seconds to perform the readiness probe. |
| api.readinessProbe.timeoutSeconds | int | `5` | The timeout in seconds for the readiness probe. |
| api.replicaCount | int | `1` | Number of replicas for the API service. |
| api.resources | object | `{}` | Kubernetes deployment resources configuration for the API service. Utilization-based autoscaling requires a matching resource request. |
| api.securityContext | object | `{}` | Container-level security context settings for the API service. |
| api.server | object | `{"keepAliveTimeoutSeconds":5}` | Platform API server settings. |
| api.server.keepAliveTimeoutSeconds | int | `5` | Seconds Uvicorn keeps idle HTTP connections open. Must be greater than envoyProxy.timeouts.upstreamIdle when Envoy is enabled. |
| api.service | object | [See values.yaml](values.yaml#L825) | Service configuration for the API service. |
| api.service.annotations | object | `{}` | Annotations for the API service. |
| api.service.port | int | `8080` | The port number to expose for the service. |
| api.service.type | string | `"ClusterIP"` | The Kubernetes service type to create. |
| api.serviceAccount | object | [See values.yaml](values.yaml#L801) | Service account configuration for the API service. |
| api.serviceAccount.annotations | object | `{}` | Annotations to add to the service account. |
| api.serviceAccount.automount | bool | `true` | Automatically mount a ServiceAccount's API credentials. |
| api.serviceAccount.create | bool | `true` | Specifies whether a service account should be created. |
| api.serviceAccount.name | string | `""` | The name of the service account to use. If not set and create is true, a name is generated using the fullname template. |
| api.serviceGroup | string | `"all"` | Predefined service group passed to `nemo services run` with `--service-group`. Ignored when api.services is non-empty. |
| api.serviceMonitor.annotations | object | `{}` | Additional annotations to add to the ServiceMonitor |
| api.serviceMonitor.enabled | bool | `false` | Enable ServiceMonitor resources for Prometheus Operator |
| api.serviceMonitor.interval | string | `"30s"` | Scrape interval for the ServiceMonitor |
| api.serviceMonitor.labels | object | `{}` | Additional labels to add to the ServiceMonitor |
| api.serviceMonitor.scheme | string | `"http"` | Scheme to use for scraping metrics (http or https) |
| api.services | list | `[]` | Explicit services passed to `nemo services run` with `--services`. When non-empty, overrides api.serviceGroup. Must be a list. |
| api.startupProbe | object | [See values.yaml](values.yaml#L837) | Startup probe configuration for the api service. |
| api.startupProbe.failureThreshold | int | `24` | The failure threshold for the startup probe. |
| api.startupProbe.httpGet | object | `{"path":"/health/ready","port":"http"}` | The HTTP GET request to use for the startup probe. |
| api.startupProbe.initialDelaySeconds | int | `10` | Number of seconds to wait before the first startup probe. Allows time for DB connection retries (e.g. Postgres pod booting). |
| api.startupProbe.periodSeconds | int | `15` | The frequency in seconds to perform the startup probe. |
| api.startupProbe.timeoutSeconds | int | `5` | The timeout in seconds for the startup probe. |
| api.telemetry | object | `{}` | OpenTelemetry configuration overrides for the api deployment. |
| api.tolerations | list | `[]` | Tolerations configuration for the API service. |
| api.topologySpreadConstraints | list | `[]` | Topology spread constraints for the API service pods. See https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/ |
| basePlatformConfig | string | [See values.yaml](values.yaml#L492) | Base platform configuration settings |
| clickhouse | object | [See values.yaml](values.yaml#L366) | Embedded ClickHouse configuration for Intake. The embedded deployment is a single-node convenience topology. Use an external, replicated ClickHouse deployment for production environments that require high availability. These values are used only when `clickhouse.enabled` is true. |
| clickhouse.affinity | object | `{}` | Affinity for the ClickHouse pod. |
| clickhouse.annotations | object | `{}` | Annotations to add to the ClickHouse StatefulSet. |
| clickhouse.auth.database | string | `"intake"` | ClickHouse database used by Intake. |
| clickhouse.auth.existingSecret | string | `""` | Name of an existing Secret containing the ClickHouse password. If empty, the chart creates one. |
| clickhouse.auth.existingSecretPasswordKey | string | `"password"` | Key in auth.existingSecret containing the ClickHouse password. |
| clickhouse.auth.password | string | `nil` | ClickHouse password used when auth.existingSecret is empty. If unset, the chart generates a random password. |
| clickhouse.auth.username | string | `"nemo"` | ClickHouse username used by Intake. |
| clickhouse.enabled | bool | `true` | Whether to deploy the embedded ClickHouse. Set to false to use `externalClickhouse`. |
| clickhouse.image.pullPolicy | string | `"IfNotPresent"` | ClickHouse image pull policy. |
| clickhouse.image.repository | string | `"docker.io/clickhouse/clickhouse-server"` | ClickHouse image repository. Intake is tested against the 26.3 LTS release line. |
| clickhouse.image.tag | string | `"26.3"` | ClickHouse image tag. |
| clickhouse.livenessProbe | object | [See values.yaml](values.yaml#L417) | Liveness probe configuration for the ClickHouse container. |
| clickhouse.nodeSelector | object | `{}` | Node selector for the ClickHouse pod. |
| clickhouse.persistence.enabled | bool | `true` | Whether to persist embedded ClickHouse data. |
| clickhouse.persistence.size | string | `"20Gi"` | PersistentVolumeClaim size. See the Intake and ClickHouse sizing guidance in the chart README. |
| clickhouse.persistence.storageClass | string | `""` | Storage class for the ClickHouse PVC. If unset, the cluster default is used. |
| clickhouse.podAnnotations | object | `{}` | Annotations to add to the ClickHouse pod. |
| clickhouse.podLabels | object | `{}` | Additional labels to add to the ClickHouse pod. |
| clickhouse.podSecurityContext | object | `{}` | Optional pod security context for the ClickHouse pod. |
| clickhouse.readinessProbe | object | [See values.yaml](values.yaml#L426) | Readiness probe configuration for the ClickHouse container. |
| clickhouse.resources | object | `{"requests":{"cpu":"2","memory":"8Gi"}}` | Resource requests and limits for the ClickHouse container. The defaults are the supported small-volume starting point. |
| clickhouse.securityContext | object | `{}` | Optional container security context for the ClickHouse container. |
| clickhouse.service.annotations | object | `{}` | Annotations to add to the ClickHouse Service. |
| clickhouse.service.httpPort | int | `8123` | ClickHouse HTTP interface port used by Intake. |
| clickhouse.service.nativePort | int | `9000` | ClickHouse native protocol port exposed inside the cluster for administration. |
| clickhouse.serviceAccount | object | [See values.yaml](values.yaml#L439) | Service account for the ClickHouse pod. |
| clickhouse.serviceAccount.annotations | object | `{}` | Annotations to add to the service account. |
| clickhouse.serviceAccount.automount | bool | `false` | Automatically mount the ServiceAccount's API credentials. |
| clickhouse.serviceAccount.create | bool | `true` | Specifies whether a service account should be created for the ClickHouse pod. |
| clickhouse.serviceAccount.name | string | `""` | The name of the service account to use. If not set and create is true, a name is generated from the release fullname. |
| clickhouse.startupProbe | object | [See values.yaml](values.yaml#L408) | Startup probe configuration for the ClickHouse container. |
| clickhouse.tolerations | list | `[]` | Tolerations for the ClickHouse pod. |
| core | object | [See values.yaml](values.yaml#L964) | Core deployment configuration settings |
| core.controller.affinity | object | `{}` | Affinity configuration for the controller service. |
| core.controller.annotations | object | `{}` | Annotations to add to the controller service deployment. |
| core.controller.controllerGroup | string | `"all"` | Predefined controller group passed to `nemo services run` with `--controller-group`. Ignored when core.controller.controllers is non-empty. |
| core.controller.controllers | list | `[]` | Explicit controllers passed to `nemo services run` with `--controllers`. When non-empty, overrides core.controller.controllerGroup. Must be a list. |
| core.controller.env | object | `{}` | Additional environment variables to pass to containers. This is an object formatted like NAME: value or NAME: valueFrom: {object}. |
| core.controller.extraArgs | list | `[]` | Additional arguments to pass to the Core Controller service |
| core.controller.livenessProbe | object | [See values.yaml](values.yaml#L1073) | Liveness probe configuration for the controller service. |
| core.controller.livenessProbe.failureThreshold | int | `3` | The failure threshold for the liveness probe. |
| core.controller.livenessProbe.httpGet | object | `{"path":"/health/live","port":"http"}` | The HTTP GET request to use for the liveness probe. |
| core.controller.livenessProbe.periodSeconds | int | `10` | The frequency in seconds to perform the liveness probe. |
| core.controller.livenessProbe.timeoutSeconds | int | `5` | The timeout in seconds for the liveness probe. |
| core.controller.nodeSelector | object | `{}` | Node selector configuration for the controller service. |
| core.controller.podAnnotations | object | `{}` | Annotations to add to the controller service pod. |
| core.controller.podLabels | object | `{}` | Labels for the controller service pod. |
| core.controller.podSecurityContext | object | [See values.yaml](values.yaml#L1047) | Pod-level security context settings for the controller service. |
| core.controller.podSecurityContext.fsGroup | int | `1000` | The file system group ID to use for all containers. |
| core.controller.readinessProbe | object | [See values.yaml](values.yaml#L1087) | Readiness probe configuration for the controller service. |
| core.controller.readinessProbe.failureThreshold | int | `3` | The failure threshold for the readiness probe. |
| core.controller.readinessProbe.httpGet | object | `{"path":"/health/ready","port":"http"}` | The HTTP GET request to use for the readiness probe. |
| core.controller.readinessProbe.periodSeconds | int | `10` | The frequency in seconds to perform the readiness probe. |
| core.controller.readinessProbe.timeoutSeconds | int | `5` | The timeout in seconds for the readiness probe. |
| core.controller.resources | object | `{}` | Kubernetes deployment resources configuration for the controller service. |
| core.controller.securityContext | object | `{}` | Container-level security context settings for the controller service. |
| core.controller.service | object | [See values.yaml](values.yaml#L1034) | Service configuration for the controller service. This only configures a headless service for DNS resolution. |
| core.controller.service.annotations | object | `{}` | Annotations for the headless controller service. |
| core.controller.service.port | int | `8080` | The port for the service. |
| core.controller.serviceAccount | object | [See values.yaml](values.yaml#L1015) | Service account configuration for the controller service. |
| core.controller.serviceAccount.annotations | object | `{}` | Annotations to add to the service account. |
| core.controller.serviceAccount.automount | bool | `true` | Automatically mount a ServiceAccount's API credentials. |
| core.controller.serviceAccount.create | bool | `true` | Specifies whether a service account should be created. |
| core.controller.serviceAccount.name | string | `""` | The name of the service account to use. If not set and create is true, a name is generated using the fullname template. |
| core.controller.startupProbe | object | [See values.yaml](values.yaml#L1057) | Startup probe configuration for the core service. |
| core.controller.startupProbe.failureThreshold | int | `24` | The failure threshold for the startup probe. |
| core.controller.startupProbe.httpGet | object | `{"path":"/health/ready","port":"http"}` | The HTTP GET request to use for the startup probe. |
| core.controller.startupProbe.initialDelaySeconds | int | `10` | Number of seconds to wait before the first startup probe. Allows time for DB connection retries (e.g. Postgres pod booting). |
| core.controller.startupProbe.periodSeconds | int | `15` | The frequency in seconds to perform the startup probe. |
| core.controller.startupProbe.timeoutSeconds | int | `5` | The timeout in seconds for the startup probe. |
| core.controller.tolerations | list | `[]` | Tolerations configuration for the controller service. |
| core.controller.topologySpreadConstraints | list | `[]` | Topology spread constraints for the controller service pods. See https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/ |
| core.enabled | bool | `true` | Specifies whether to enable the core deployment. |
| core.image | object | [See values.yaml](values.yaml#L970) | Container image configuration for the core deployment. |
| core.image.pullPolicy | string | `"IfNotPresent"` | The image pull policy determining when to pull new images. |
| core.image.repository | string | `"nvcr.io/nvidia/nemo-platform/nmp-api"` | The registry where the NeMo Platform image is located. |
| core.image.tag | string | `""` | The image tag to use. |
| core.jobs | object | [See values.yaml](values.yaml#L1000) | Service account configuration for pods created by the jobs controller (Kubernetes/Volcano job pods). |
| core.jobs.serviceAccount.annotations | object | `{}` | Annotations to add to the service account. |
| core.jobs.serviceAccount.automount | bool | `true` | Automatically mount a ServiceAccount's API credentials. |
| core.jobs.serviceAccount.create | bool | `true` | Specifies whether a service account should be created for job pods. |
| core.jobs.serviceAccount.name | string | `""` | The name of the service account to use. If not set and create is true, a name is generated with a '-jobs' suffix. |
| core.serviceMonitor.annotations | object | `{}` | Additional annotations to add to the ServiceMonitor |
| core.serviceMonitor.enabled | bool | `false` | Enable ServiceMonitor resources for Prometheus Operator |
| core.serviceMonitor.interval | string | `"30s"` | Scrape interval for the ServiceMonitor |
| core.serviceMonitor.labels | object | `{}` | Additional labels to add to the ServiceMonitor |
| core.serviceMonitor.scheme | string | `"http"` | Scheme to use for scraping metrics (http or https) |
| core.storage.accessModes | list | `["ReadWriteMany"]` | accessModes for the persistent volume claim. This should include `ReadWriteMany` to ensure multiple job pods can write to the volume concurrently. |
| core.storage.annotations | object | `{}` | Annotations to add to the persistent volume claim |
| core.storage.existingPersistentVolumeName | string | `""` | If set, pods will mount this persistent volume for job-scoped storage and we will not create a new persistent volume claim. |
| core.storage.size | string | `"200Gi"` | size of the persistent volume claim used for persistent storage |
| core.storage.storageClass | string | `""` | Which storageClass to use when creating a new persistent volume claim. Empty string uses the cluster's default StorageClass. |
| core.storage.volumePermissionsImage | string | `"docker.io/library/busybox:stable"` | volumePermissionsImage is the image used to set permissions on the volume |
| core.telemetry | object | `{}` | OpenTelemetry configuration overrides for the platform deployment. |
| env | object | `{}` | Environment variables that will be applied to every deployment pod. Uses a simple key value map structure like MY_ENV_VAR: the-key and works with valueFrom as well. |
| envFromSecret | string | `""` | Optional. Name of an existing Kubernetes Secret to load as env vars (envFrom) for the API pod. When set, the chart does not create or generate the default api-env Secret; use your own Secret (for example, from Vault or sealed-secrets). |
| envoyProxy | object | [See values.yaml](values.yaml#L1125) | Envoy proxy configuration settings. Resources are created only when platform config has auth.enabled: true (see platformConfig.auth.enabled). |
| envoyProxy.adminPort | int | `9901` | Envoy Admin port |
| envoyProxy.affinity | object | `{}` | Affinity configuration for the Envoy pods. |
| envoyProxy.annotations | object | `{}` | Annotations to add to the Envoy service deployment. |
| envoyProxy.autoscaling | object | [See values.yaml](values.yaml#L1250) | Specifies autoscaling configurations for the deployment. |
| envoyProxy.autoscaling.annotations | object | `{}` | Annotations for the HorizontalPodAutoscaler. |
| envoyProxy.autoscaling.enabled | bool | `false` | Whether to enable horizontal pod autoscaler. |
| envoyProxy.autoscaling.maxReplicas | int | `10` | The maximum number of replicas for the deployment. |
| envoyProxy.autoscaling.minReplicas | int | `1` | The minimum number of replicas for the deployment. |
| envoyProxy.autoscaling.targetCPUUtilizationPercentage | int | `80` | The target CPU utilization percentage. Requires envoyProxy.resources.requests.cpu. |
| envoyProxy.autoscaling.targetMemoryUtilizationPercentage | string | `nil` | The target memory utilization percentage. Requires envoyProxy.resources.requests.memory. |
| envoyProxy.configOverride | string | `""` | Full Envoy config override. When set, this replaces the chart's default passthrough Envoy config. |
| envoyProxy.enabled | bool | `true` | Specifies whether to enable the Envoy proxy deployment. Rendered only when platform config has auth.enabled: true. |
| envoyProxy.extraArgs | list | `[]` | Extra arguments to append to the envoy container command. Useful for passing server flags such as concurrency. Example: ["--concurrency", "4"] |
| envoyProxy.extraVolumeMounts | list | `[]` | Additional volume mounts to add to the Envoy container. |
| envoyProxy.extraVolumes | list | `[]` | Additional volumes to add to the Envoy pod. |
| envoyProxy.image.digest | string | `""` | Optional image digest. When set, the Envoy image renders as repository@digest. |
| envoyProxy.livenessProbe | object | [See values.yaml](values.yaml#L1211) | Liveness probe for the Envoy container (admin interface /ready). |
| envoyProxy.nodeSelector | object | `{}` | Node selector configuration for the Envoy pods. |
| envoyProxy.podAnnotations | object | `{}` | Annotations to add to the Envoy service pod. |
| envoyProxy.podDisruptionBudget | object | [See values.yaml](values.yaml#L1237) | PodDisruptionBudget configuration for the Envoy service. |
| envoyProxy.podDisruptionBudget.annotations | object | `{}` | Annotations for the PodDisruptionBudget. |
| envoyProxy.podDisruptionBudget.enabled | bool | `false` | Whether to create a PodDisruptionBudget for the Envoy pods. |
| envoyProxy.podDisruptionBudget.minAvailable | int | `1` | Minimum number of Envoy pods that must remain available during voluntary disruptions. Only one of minAvailable or maxUnavailable may be set. |
| envoyProxy.podLabels | object | `{}` | Labels for the Envoy service pod. |
| envoyProxy.podSecurityContext | object | [See values.yaml](values.yaml#L1171) | Pod-level security context settings for the Envoy service. |
| envoyProxy.podSecurityContext.fsGroup | int | `1000` | The file system group ID to use for all containers. |
| envoyProxy.readinessProbe | object | [See values.yaml](values.yaml#L1219) | Readiness probe for the Envoy container (admin interface /ready). |
| envoyProxy.resources | object | `{}` | Kubernetes deployment resources configuration for the Envoy service. Utilization-based autoscaling requires a matching resource request. |
| envoyProxy.securityContext | object | `{}` | Container-level security context settings for the Envoy service. |
| envoyProxy.service | object | [See values.yaml](values.yaml#L1178) | Service configuration for the Envoy service. |
| envoyProxy.service.annotations | object | `{}` | Annotations for the Envoy service. |
| envoyProxy.service.exposeAdminPort | bool | `false` | Expose the Envoy admin port through the Kubernetes Service. Enable only for controlled in-cluster scraping or debugging. |
| envoyProxy.service.port | int | `8080` | The port number to expose for the service. |
| envoyProxy.service.type | string | `"ClusterIP"` | The Kubernetes service type to create. |
| envoyProxy.serviceAccount | object | [See values.yaml](values.yaml#L1154) | Service account configuration for the Envoy service. |
| envoyProxy.serviceAccount.annotations | object | `{}` | Annotations to add to the service account. |
| envoyProxy.serviceAccount.automount | bool | `true` | Automatically mount a ServiceAccount's API credentials. |
| envoyProxy.serviceAccount.create | bool | `true` | Specifies whether a service account should be created. |
| envoyProxy.serviceAccount.name | string | `""` | The name of the service account to use. If not set and create is true, a name is generated using the fullname template. |
| envoyProxy.serviceMonitor.annotations | object | `{}` | Additional annotations to add to the ServiceMonitor |
| envoyProxy.serviceMonitor.enabled | bool | `false` | Enable ServiceMonitor resources for Prometheus Operator |
| envoyProxy.serviceMonitor.interval | string | `"30s"` | Scrape interval for the ServiceMonitor |
| envoyProxy.serviceMonitor.labels | object | `{}` | Additional labels to add to the ServiceMonitor |
| envoyProxy.serviceMonitor.scheme | string | `"http"` | Scheme to use for scraping metrics (http or https) |
| envoyProxy.startupProbe | object | [See values.yaml](values.yaml#L1227) | Startup probe for the Envoy container (admin interface /ready). |
| envoyProxy.timeouts | object | [See values.yaml](values.yaml#L1193) | Timeouts for proxying to long-lived streams (e.g. inference gateway). Use "0s" to disable a timeout. |
| envoyProxy.timeouts.connect | string | `"30s"` | Cluster connect timeout (time to establish connection to backend). |
| envoyProxy.timeouts.request | string | `"0s"` | Total request timeout. 0 = disabled (required for streaming; not compatible with streaming if set). |
| envoyProxy.timeouts.requestHeaders | string | `"60s"` | Time to receive full request headers. 0 = disabled. |
| envoyProxy.timeouts.route | string | `"0s"` | Per-route timeout for the passthrough to backend. 0 = disabled. |
| envoyProxy.timeouts.streamIdle | string | `"0s"` | Stream idle timeout. Time with no activity before stream is closed. 0 = disabled (required for long-lived streams). |
| envoyProxy.timeouts.upstreamIdle | string | `"4s"` | Positive whole-second upstream connection idle timeout. Must be less than api.server.keepAliveTimeoutSeconds when Envoy is enabled. |
| envoyProxy.tolerations | list | `[]` | Tolerations configuration for the Envoy pods. |
| envoyProxy.topologySpreadConstraints | list | `[]` | Topology spread constraints for the Envoy pods. See https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/ |
| existingSecret | string | `"ngc-api"` | You can use an existing Kubernetes secret for communicating with the NGC API for downloading models. The chart uses the `ngcAPIKey` value to generate the secret if you set this to an empty string. |
| externalClickhouse | object | [See values.yaml](values.yaml#L463) | External ClickHouse configuration. These values are used when `clickhouse.enabled` is false. |
| externalClickhouse.database | string | `"intake"` | ClickHouse database used by Intake. |
| externalClickhouse.existingSecret | string | `""` | Name of an existing Secret containing the ClickHouse password. Required when using external ClickHouse. |
| externalClickhouse.existingSecretPasswordKey | string | `""` | Key in existingSecret containing the ClickHouse password. Required when using external ClickHouse. |
| externalClickhouse.host | string | `""` | External ClickHouse host. Required when the embedded ClickHouse is disabled. |
| externalClickhouse.port | int | `8123` | External ClickHouse HTTP interface port. |
| externalClickhouse.secure | bool | `false` | Whether Intake should connect to ClickHouse over HTTPS. |
| externalClickhouse.user | string | `"nemo"` | ClickHouse username used by Intake. |
| externalDatabase | object | [See values.yaml](values.yaml#L340) | External PostgreSQL configuration settings. These values are only used when postgresql.enabled is set to false. |
| externalDatabase.database | string | `"nemoplatform"` | Database name. |
| externalDatabase.existingSecret | string | `""` | Name of an existing secret resource containing the database credentials. |
| externalDatabase.existingSecretPasswordKey | string | `""` | Name of an existing secret key containing the database credentials. |
| externalDatabase.host | string | `"localhost"` | External database host address. |
| externalDatabase.port | int | `5432` | External database port number. |
| externalDatabase.uriSecret | object | [See values.yaml](values.yaml#L355) | URI secret configuration for external database. |
| externalDatabase.uriSecret.key | string | `""` | Key in the URI secret containing the database URI. |
| externalDatabase.uriSecret.name | string | `""` | Name of the URI secret. |
| externalDatabase.user | string | `"nemo"` | Database username |
| fullnameOverride | string | `""` |  |
| httpRoute.annotations | object | `{}` | Extra annotations for the HTTP Route object. |
| httpRoute.enabled | bool | `false` | Specifies whether to enable a Gateway API HTTP Route for the service. |
| httpRoute.filters | list | `[]` | This is a list of filters for the objects, such as CORS settings. |
| httpRoute.hostnames | list | `[]` | If this has a specific hostname, add the name or names here in an array. |
| httpRoute.labels | object | `{}` | Extra labels for the HTTP Route object. |
| httpRoute.parentRefs | list | `[]` | A list of Gateways to enable this route on. This is required if httpRoute.enabled is true. |
| httpRoute.pathRules | list | [See values.yaml](values.yaml#L696) | Path matches to route queries. |
| imagePullSecrets | list | `[]` | Existing Kubernetes image pull secrets to use for pulling container images from private registries or mirrors. |
| ingress.annotations | object | `{}` | Annotations for the ingress resource. |
| ingress.className | string | `""` | The ingress class to use if your cluster has more than one class. |
| ingress.defaultHost | string | `""` | Optional default hostname. When set, one rule is generated with this host and paths from the first entry in ingress.hosts. |
| ingress.enabled | bool | `false` | Specifies whether to enable the ingress. |
| ingress.hosts[0] | object | [See values.yaml](values.yaml#L661) | Hostname used by ingress. If blank, use path-only routing. |
| ingress.tls | list | `[]` | TLS configurations. |
| multinodeNetworking | object | [See values.yaml](values.yaml#L112) | Multi-node networking configuration for distributed GPU training. These settings control Kyverno policies that inject cloud-specific networking and NCCL configurations.  Requirements: - Kyverno policy engine must be installed in your cluster (required for multi-node networking) - Kyverno is NOT included as a subchart dependency and must be installed separately  To install Kyverno:   helm install kyverno kyverno/kyverno --namespace kyverno --create-namespace --version 3.2.0  Documentation: https://kyverno.io/docs/installation/ Helm chart: https://kyverno.github.io/kyverno/  Note: Only enable ONE cloud provider per cluster deployment. |
| multinodeNetworking.aws | object | `{"efaDevicesPerGPU":1,"enabled":false}` | AWS-specific configuration for EFA device injection |
| multinodeNetworking.aws.efaDevicesPerGPU | int | `1` | Number of EFA devices to request per GPU (typically 1 or 4) |
| multinodeNetworking.aws.enabled | bool | `false` | Enable AWS-specific Kyverno policy for EFA device injection |
| multinodeNetworking.azure | object | [See values.yaml](values.yaml#L121) | Azure-specific configuration for InfiniBand/RDMA |
| multinodeNetworking.azure.enabled | bool | `false` | Enable Azure-specific Kyverno policy for InfiniBand/RDMA configuration |
| multinodeNetworking.azure.rdmaDeviceName | string | `"hca_shared_devices_a"` | RDMA device plugin resource name |
| multinodeNetworking.azure.rdmaDevicesPerGPU | int | `1` | Number of RDMA devices to request per GPU |
| multinodeNetworking.gcp | object | `{"enabled":false}` | GCP-specific configuration for TCP-X/TCP-XO |
| multinodeNetworking.gcp.enabled | bool | `false` | Enable GCP-specific Kyverno policy for TCP-X/TCP-XO configuration |
| multinodeNetworking.oci | object | `{"enabled":false,"rdmaDevicesPerGPU":8}` | OCI-specific configuration for InfiniBand/SR-IOV |
| multinodeNetworking.oci.enabled | bool | `false` | Enable OCI-specific Kyverno policy for InfiniBand/SR-IOV configuration |
| multinodeNetworking.oci.rdmaDevicesPerGPU | int | `8` | Number of RDMA devices (mlnxnics) to request per GPU |
| nameOverride | string | `""` | Overrides for name and fullname templates |
| ncclTest | object | [See values.yaml](values.yaml#L239) | NCCL chart test (`helm test`): multi-node allreduce check. Templates use helm.sh/hook: test — they are not created on install/upgrade, only when you run helm test. Requires nodes labeled with gpuNodeLabelKey/gpuNodeLabelValue (default NFD / GPU operator style). See https://helm.sh/docs/topics/chart_tests/ |
| ncclTest.configMapCleanupJob | object | [See values.yaml](values.yaml#L268) | Post-test hook Job (after orchestrator): deletes the scripts ConfigMap (helm.sh/hook-weight 5). |
| ncclTest.gpuNodeLabelKey | string | `"nvidia.com/gpu.present"` | Node label used to discover GPU workers (must match your cluster). |
| ncclTest.gpuResourceKey | string | `"nvidia.com/gpu"` | Resource name for GPU capacity on worker pods (e.g. nvidia.com/gpu or a MIG device). |
| ncclTest.gpusPerNode | int | `1` | GPUs per worker pod / per node (torch.distributed nproc_per_node). IMPORTANT: Set this value before testing |
| ncclTest.iterations | int | `3` | How many times to run the full multinode NCCL test (orchestrator loop; env NCCL_TEST_ITERATIONS). Increase the test timeout on helm test if increasing this variable |
| ncclTest.validation.minBandwidthMBpsAt1024MB | int | `8000` | Minimum allreduce bandwidth (MB/s) at 1024MB message size; 0 disables the floor check in nccl_test.py. |
| ncclTest.waitTimeoutSeconds | int | `900` | Max seconds to wait for each worker pod to complete. |
| networkPolicies | object | [See values.yaml](values.yaml#L143) | NetworkPolicy configuration. Enable the top-level switch to render all default policies, then disable individual policies only for cluster-specific exceptions. For a Calico-backed smoke test, see https://docs.nvidia.com/nemo-platform/documentation/kubernetes-deployment/setup/helm/network-policy-smoke-test. |
| networkPolicies.api | object | [See values.yaml](values.yaml#L148) | NetworkPolicy configuration for the Platform API pods. |
| networkPolicies.api.enabled | bool | `true` | Create NetworkPolicy resources that isolate Platform API pod ingress. |
| networkPolicies.api.extraIngress | list | `[]` | Extra NetworkPolicy ingress rules appended to the API policy, for cluster-specific ingress controllers, gateways, monitoring, or debugging pods. |
| networkPolicies.api.managedJobs | object | [See values.yaml](values.yaml#L158) | Allow managed job pods to reach the Platform API pods. |
| networkPolicies.api.managedJobs.enabled | bool | `true` | Enable ingress from managed job pods. |
| networkPolicies.api.managedJobs.podSelector | object | [See values.yaml](values.yaml#L162) | Pod selector for managed job pods. The default matches Kubernetes/Volcano pods created by the jobs controller. |
| networkPolicies.api.port | string | `""` | Optional NetworkPolicy port for the Platform API. Empty uses api.service.port. |
| networkPolicies.api.sameReleasePods | object | `{"enabled":true}` | Allow pods from the same Helm release namespace that carry the chart selector labels to reach the Platform API pods. |
| networkPolicies.api.sameReleasePods.enabled | bool | `true` | Enable ingress from same-release pods. |
| networkPolicies.controller | object | [See values.yaml](values.yaml#L171) | NetworkPolicy configuration for the core controller pods. |
| networkPolicies.controller.enabled | bool | `true` | Create NetworkPolicy resources that isolate core controller pod ingress. |
| networkPolicies.controller.extraIngress | list | `[]` | Extra NetworkPolicy ingress rules appended to the core controller policy, for cluster-specific monitoring or debugging pods. |
| networkPolicies.controller.port | string | `""` | Optional NetworkPolicy port for the core controller. Empty uses core.controller.service.port. |
| networkPolicies.controller.sameReleasePods | object | `{"enabled":true}` | Allow pods from the same Helm release namespace that carry the chart selector labels to reach the core controller pods. |
| networkPolicies.controller.sameReleasePods.enabled | bool | `true` | Enable ingress from same-release pods. |
| networkPolicies.enabled | bool | `false` | Create NetworkPolicy resources for enabled subpolicies. |
| networkPolicies.jobs | object | [See values.yaml](values.yaml#L185) | NetworkPolicy configuration for pods created by the jobs controller. |
| networkPolicies.jobs.dns | object | [See values.yaml](values.yaml#L200) | Allow DNS lookup from managed job pods. |
| networkPolicies.jobs.dns.enabled | bool | `true` | Enable egress to DNS pods. |
| networkPolicies.jobs.dns.namespaceSelector | object | [See values.yaml](values.yaml#L204) | Namespace selector for DNS pods. |
| networkPolicies.jobs.dns.podSelector | object | `{"matchLabels":{"k8s-app":"kube-dns"}}` | Pod selector for DNS pods. |
| networkPolicies.jobs.dns.ports | list | [See values.yaml](values.yaml#L212) | DNS ports to allow. |
| networkPolicies.jobs.enabled | bool | `true` | Create NetworkPolicy resources that isolate managed job pod egress. |
| networkPolicies.jobs.externalEgress | object | [See values.yaml](values.yaml#L218) | Allow egress to external CIDR blocks. The default excludes common private, loopback, and link-local ranges so enabling the policy does not allow direct access to typical in-cluster service and pod networks. |
| networkPolicies.jobs.externalEgress.enabled | bool | `true` | Enable egress to external CIDR blocks. |
| networkPolicies.jobs.externalEgress.ipBlocks | list | [See values.yaml](values.yaml#L222) | External destination CIDR blocks for managed job pods. |
| networkPolicies.jobs.externalEgress.ports | list | `[]` | Optional list of ports for external egress. Empty allows all ports to the configured ipBlocks. |
| networkPolicies.jobs.extraEgress | list | `[]` | Extra NetworkPolicy egress rules appended to the managed job policy, for cluster-specific dependencies or additional allowed CIDRs. |
| networkPolicies.jobs.platformApi | object | `{"enabled":true,"port":""}` | Allow managed job pods to reach the in-namespace Platform API pods. |
| networkPolicies.jobs.platformApi.enabled | bool | `true` | Enable the default egress exception to the platform API pods. Disable only when managed job pods should not call the in-namespace API, or when API access is supplied through cluster-specific extraEgress rules. |
| networkPolicies.jobs.platformApi.port | string | `""` | Optional NetworkPolicy port for the platform API. Empty uses api.service.port. |
| networkPolicies.jobs.podSelector | object | [See values.yaml](values.yaml#L189) | Pod selector for managed job pods. The default matches Kubernetes/Volcano pods created by the jobs controller. |
| ngcAPIKey | string | `"YOUR-NGC-API-KEY"` | Your NVIDIA GPU Cloud (NGC) API key authenticates API calls to NGC services, such as model downloads. The existing secret overrides this key if you provide one to the `existingSecret` key. |
| opensandbox | object | [See values.yaml](values.yaml#L83) | Connection settings for an already-installed OpenSandbox server. Used when sandboxClusterCapable is true. Copy the API-key Secret into the Helm release namespace (the namespace jobs run in). Override domain for a Kata Service. |
| opensandbox.apiKeySecret | string | `"opensandbox-server-api-key"` | Kubernetes Secret in the job/release namespace holding the API key. |
| opensandbox.apiKeySecretKey | string | `"api-key"` | Key inside apiKeySecret. |
| opensandbox.domain | string | [See values.yaml](values.yaml#L85) | In-cluster OpenSandbox Service DNS with no scheme. |
| opensandbox.protocol | string | `"http"` | Scheme jobs use to reach the server. In-cluster Services speak http. |
| openshiftRoute | object | [See values.yaml](values.yaml#L715) | OpenShift Route (route.openshift.io/v1). Use on OpenShift to expose the API via a Route instead of Ingress. |
| openshiftRoute.annotations | object | `{}` | Annotations for the route resource. |
| openshiftRoute.enabled | bool | `false` | Specifies whether to create an OpenShift Route for the API service. |
| openshiftRoute.host | string | `""` | Hostname for the route. If empty, the OpenShift router may assign a default hostname. |
| openshiftRoute.labels | object | `{}` | Labels for the route resource. |
| openshiftRoute.service | string | [See values.yaml](values.yaml#L721) | Service name to route to. Defaults to Envoy when auth+envoy enabled, otherwise API (tpl-evaluated). |
| openshiftRoute.targetPort | string | [See values.yaml](values.yaml#L723) | Target port on the service. Defaults to Envoy or API port depending on auth (tpl-evaluated). |
| openshiftRoute.tls | object | `{}` | Optional TLS configuration (termination, certificate, key, etc.). See OpenShift Route spec. |
| platformConfig | object | [See values.yaml](values.yaml#L485) | Platform-wide configuration settings Set configuration here to apply custom, structured configuration across all services. Applied after the base platform config is evaluated for templates. Enables adding / overriding YAML-based elements in the evaluated platform config. It is usually recommended to use this config section instead of `basePlatformConfig` unless you need to use templating features. For example, you can set the NIM default StorageClass via models.controller.backends.deployments_plugin.default_storage_class. For full configuration reference, see https://docs.nvidia.com/nemo-platform |
| platformSeedJob | object | [See values.yaml](values.yaml#L936) | Platform seed Job (Helm hook: runs after install/upgrade) Runs the platform-seed task (guardrails configs, evaluator system entities, data designer filesets). Uses post-install,post-upgrade hooks so it runs on fresh installs and can be re-triggered on no-op upgrade. |
| platformSeedJob.activeDeadlineSeconds | int | `600` | Maximum time in seconds the Job can run. |
| platformSeedJob.affinity | object | `{}` | Affinity for the platform seeding Job pod. |
| platformSeedJob.backoffLimit | int | `6` | Number of retries before considering the Job failed. |
| platformSeedJob.enabled | bool | `true` | Specifies whether to enable the platform-seed Job. |
| platformSeedJob.extraEnv | list | `[]` | Extra environment variables for the platform-seed container (e.g. CONFIG_STORE_PATH, NMP_PLATFORM_SEED_*). |
| platformSeedJob.nodeSelector | object | `{}` | Node selector for the platform seeding Job pod. |
| platformSeedJob.podLabels | object | `{}` | Additional labels for the platform seeding Job pod. |
| platformSeedJob.podSecurityContext | object | `{}` | Pod-level security context for the platform seeding Job pod. |
| platformSeedJob.resources | object | `{}` | Resource requests/limits for the platform-seed container. |
| platformSeedJob.securityContext | object | `{}` | Container-level security context for the platform-seed container. |
| platformSeedJob.tolerations | list | `[]` | Tolerations for the platform seeding Job pod. |
| platformSeedJob.ttlSecondsAfterFinished | int | `86400` | Seconds after the Job finishes (success or failure) before it is eligible for automatic deletion. |
| podSecurityContext | object | [See values.yaml](values.yaml#L756) | Pod security context settings applied to all services by default. These can be overridden in individual service configurations. |
| postgresql | object | [See values.yaml](values.yaml#L286) | Local PostgreSQL configuration for the NeMo Platform. |
| postgresql.affinity | object | `{}` | Affinity for the PostgreSQL pod. |
| postgresql.auth | object | [See values.yaml](values.yaml#L296) | PostgreSQL authentication configuration. |
| postgresql.auth.existingSecret | string | `""` | Name of an existing secret containing a "password" key (or use existingSecretPasswordKey). If set, the chart does not create a secret. |
| postgresql.enabled | bool | `true` | Whether to deploy the embedded PostgreSQL. If enabled, the chart deploys a single-replica PostgreSQL instance using the official Postgres image. It is NOT recommended to use the built-in PostgreSQL for production deployments. It is enabled in the chart by default for ease of getting started with the platform. If you are using an existing PostgreSQL installation, set this to false and use the "externalDatabase" configuration section. |
| postgresql.nodeSelector | object | `{}` | Node selector for the PostgreSQL pod. |
| postgresql.persistence | object | `{"enabled":true,"size":"5Gi","storageClass":""}` | PostgreSQL persistence configuration. |
| postgresql.persistence.storageClass | string | `""` | Storage class for the PostgreSQL PVC. If unset, the cluster default is used. |
| postgresql.podSecurityContext | object | `{}` | Optional pod security context for the PostgreSQL pod (e.g. for OpenShift SCC). |
| postgresql.resources | object | `{}` | Optional resource limits/requests for the PostgreSQL container. |
| postgresql.securityContext | object | `{}` | Optional container security context for the PostgreSQL container. |
| postgresql.service | object | `{"port":5432}` | PostgreSQL service configuration. |
| postgresql.serviceAccount | object | [See values.yaml](values.yaml#L322) | Service account for the PostgreSQL pod. |
| postgresql.serviceAccount.annotations | object | `{}` | Annotations to add to the service account. |
| postgresql.serviceAccount.automount | bool | `true` | Automatically mount the ServiceAccount's API credentials. |
| postgresql.serviceAccount.create | bool | `true` | Specifies whether a service account should be created for the PostgreSQL pod. |
| postgresql.serviceAccount.name | string | `""` | The name of the service account to use. If not set and create is true, a name is generated from the release fullname. |
| postgresql.tolerations | list | `[]` | Tolerations for the PostgreSQL pod. |
| rbac | object | `{"volcanoEnabled":true}` | RBAC configuration settings for optional dependencies |
| rbac.volcanoEnabled | bool | `true` | Specifies whether to enable the core Controller to have RBAC permissions to Volcano for scheduling distributed jobs. |
| sandboxClusterCapable | bool | `false` | Whether OpenSandbox is installed on this cluster. The chart does not install OpenSandbox. Defaults to false so sandboxed GRPO fail-closes until you deploy OpenSandbox and set this true. See the OpenSandbox section in this README. |
| secrets | object | [See values.yaml](values.yaml#L23) | Secrets service configuration. |
| secrets.defaultEncryptionKey.existingSecret | object | [See values.yaml](values.yaml#L28) | Existing Kubernetes Secret containing the key for encrypting platform secrets. If name is set, the chart does not create or generate the default api-env Secret. |
| secrets.defaultEncryptionKey.existingSecret.key | string | `"NMP_SECRETS_DEFAULT_ENCRYPTION_KEY"` | Key in the existing Secret. After Kubernetes decodes the Secret data, the loaded value must be the base64-encoded NMP_SECRETS_DEFAULT_ENCRYPTION_KEY string. |
| secrets.defaultEncryptionKey.existingSecret.name | string | `""` | Name of an existing Kubernetes Secret containing the encryption key. |
| secrets.defaultEncryptionKey.generated | object | [See values.yaml](values.yaml#L34) | Generated key configuration used only when value and envFromSecret are empty. The generated key is not rotated or recreated on upgrade. |
| secrets.defaultEncryptionKey.generated.activeDeadlineSeconds | int | `120` | Maximum seconds for the key generation hook to run. |
| secrets.defaultEncryptionKey.generated.affinity | object | `{}` | Affinity for the key generation hook. |
| secrets.defaultEncryptionKey.generated.backoffLimit | int | `3` | Number of retries before the key generation hook is marked failed. |
| secrets.defaultEncryptionKey.generated.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy for the pre-install key generation hook. |
| secrets.defaultEncryptionKey.generated.image.repository | string | `"docker.io/library/python"` | Image repository for the pre-install key generation hook. |
| secrets.defaultEncryptionKey.generated.image.tag | string | `"3.12-slim"` | Image tag for the pre-install key generation hook. |
| secrets.defaultEncryptionKey.generated.nodeSelector | object | `{}` | Node selector for the key generation hook. |
| secrets.defaultEncryptionKey.generated.podSecurityContext | object | `{}` | Optional pod security context for the key generation hook. |
| secrets.defaultEncryptionKey.generated.resources | object | `{}` | Optional resource limits/requests for the key generation hook. |
| secrets.defaultEncryptionKey.generated.securityContext | object | `{}` | Optional container security context for the key generation hook. |
| secrets.defaultEncryptionKey.generated.serviceAccount.annotations | object | `{}` | Annotations to add to the key generation hook service account. |
| secrets.defaultEncryptionKey.generated.serviceAccount.create | bool | `true` | Specifies whether a service account should be created for the key generation hook. |
| secrets.defaultEncryptionKey.generated.serviceAccount.name | string | `""` | The name of the service account to use. Required when create is false. If not set and create is true, a name is generated using the fullname template. |
| secrets.defaultEncryptionKey.generated.tolerations | list | `[]` | Tolerations for the key generation hook. |
| secrets.defaultEncryptionKey.generated.ttlSecondsAfterFinished | int | `300` | Seconds to keep the key generation hook Job after it finishes, if the hook is not deleted first. |
| secrets.defaultEncryptionKey.value | string | `""` | Optional base64-encoded key for encrypting platform secrets. The decoded key must be at least 32 bytes. If empty and envFromSecret is not set, a pre-install hook generates a per-install key. |
| securityContext | object | [See values.yaml](values.yaml#L761) | Container security context settings applied to all services by default. These can be overridden in individual service configurations. |
| telemetry.OTEL_EXPORTER_OTLP_ENDPOINT | string | `""` | The OpenTelemetry grpc collector endpoint to export traces and metrics to. |
| telemetry.OTEL_EXPORTER_OTLP_INSECURE | bool | `true` | Whether to use an insecure connection (no TLS) to the OpenTelemetry collector endpoint. |
| telemetry.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT | string | `nil` | The OpenTelemetry metrics exporter endpoint to use. Defaults to `OTEL_EXPORTER_OTLP_ENDPOINT` if not set. |
| telemetry.OTEL_EXPORTER_OTLP_METRICS_INSECURE | bool | `true` | Whether to use an insecure connection (HTTP) to the OpenTelemetry metrics exporter endpoint. Defaults to `OTEL_EXPORTER_OTLP_INSECURE` if not set. |
| telemetry.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT | string | `nil` | The OpenTelemetry traces exporter endpoint to use. Defaults to `OTEL_EXPORTER_OTLP_ENDPOINT` if not set. |
| telemetry.OTEL_EXPORTER_OTLP_TRACES_INSECURE | bool | `true` | Whether to use an insecure connection (HTTP) to the OpenTelemetry traces exporter endpoint. Defaults to `OTEL_EXPORTER_OTLP_INSECURE` if not set. |
| telemetry.OTEL_METRICS_EXPORTER | string | `"none"` | The OpenTelemetry metrics exporter to use. Options are "otlp", "prometheus" or "none" to disable export. |
| telemetry.OTEL_SDK_DISABLED | bool | `false` | Disable OpenTelemetry instrumentation and exporting for all services. |
| telemetry.OTEL_TRACES_EXPORTER | string | `"none"` | The OpenTelemetry traces exporter to use. Options are "otlp" or "none" to disable export. |
