{{/*
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{/*
The namespace builds run in.
*/}}
{{- define "nemo-helix.builder.namespace" -}}
{{- default (printf "%s-builds" .Release.Namespace) .Values.builder.namespace | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
The build steps' ServiceAccount, in the build namespace. Takes (list $ "<step>"), the step being fetch, control or push.
*/}}
{{- define "nemo-helix.builder.serviceAccountName" -}}
{{- printf "nhx-build-%s" (index . 1) -}}
{{- end -}}

{{/*
The work volume, in the build namespace.
*/}}
{{- define "nemo-helix.builder.workVolume" -}}
nhx-build-work
{{- end -}}

{{/*
The development registry's Service, in the release namespace.
*/}}
{{- define "nemo-helix.builder.devRegistryName" -}}
{{- printf "%s-builder-registry" (include "nemo-helix.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
What the chart sets in the platform config when the builder is enabled. The user's platformConfig is merged over it.
Build pods run in another namespace, so the platform's URL they get must be namespace-qualified.
*/}}
{{- define "nemo-helix.builder.platformConfig" -}}
{{- if .Values.builder.enabled }}
platform:
  base_url: {{ printf "http://%s.%s.svc:%s" (include "nhx-api.api-servicename" .) .Release.Namespace (toString .Values.api.service.port) | quote }}
{{- if .Values.builder.devRegistry.enabled }}
builder:
  # Fully qualified: registry clients, cosign's among them, speak plain HTTP by default only to a `.local` host.
  registry: {{ printf "http://%s.%s.svc.cluster.local:5000" (include "nemo-helix.builder.devRegistryName" .) .Release.Namespace | quote }}
  {{- end }}
{{- end }}
{{- end -}}

{{/*
The builder's Jobs execution profiles, one per step, all in the build namespace on one work volume. Rendered in full:
a profile the config adds doesn't inherit executor_defaults.
*/}}
{{- define "nemo-helix.builder.executors" -}}
{{- if .Values.builder.enabled }}
{{- range $step := list "fetch" "control" "push" }}
- provider: cpu
  profile: {{ printf "build-%s" $step }}
  backend: kubernetes_job
  config:
    namespace: {{ include "nemo-helix.builder.namespace" $ | quote }}
    service_account_name: {{ include "nemo-helix.builder.serviceAccountName" (list $ $step) | quote }}
    launcher_image: {{ include "nhx-core.image" $ | quote }}
    # Deletes each finished step's Job and, once the job succeeds, its directory on the work volume. Nothing else does.
    cleanup_completed_jobs_immediately: true
    storage:
      pvc_name: {{ include "nemo-helix.builder.workVolume" $ | quote }}
      volume_permissions_image: {{ $.Values.core.storage.volumePermissionsImage | quote }}
    pod_security_context: {{ $.Values.podSecurityContext | toJson }}
{{- end }}
{{- end }}
{{- end -}}
