{{/*
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{/*
Image Definition Parsing
Favor not using a separate registry because it is confusing, but support it.
*/}}
{{- define "nhx-core.image" -}}
{{- if .Values.core.image.registry -}}
{{ .Values.core.image.registry }}/{{ .Values.core.image.repository }}:{{ default .Chart.AppVersion .Values.core.image.tag }}
{{- else -}}
{{ .Values.core.image.repository }}:{{ default .Chart.AppVersion .Values.core.image.tag }}
{{- end }}
{{- end }}

{{/*
Create a named core service name which can be included from parent chart
*/}}
{{- define "nhx-core.api-servicename" }}
{{- printf "%s-core" ( include "nemo-helix.fullname" . | trunc 59 ) }}
{{- end }}

{{/*
Create a named core controller service name which can be included from parent chart
*/}}
{{- define "nhx-core.controller-servicename" }}
{{- printf "%s-core-controller" ( include "nemo-helix.fullname" . | trunc 52 ) }}
{{- end }}

{{/*
Create a named core controller service name which can be included from parent chart
*/}}
{{- define "nhx-core.database-migrations-servicename" }}
{{- printf "%s-core-migrations" ( include "nemo-helix.fullname" .) }}
{{- end }}

{{/*
Create the name of the API service account to use
*/}}
{{- define "nhx-core.apiServiceAccountName" -}}
{{- if .Values.core.api.serviceAccount.create }}
{{- default (printf "%s-core" (include "nemo-helix.fullname" .)) .Values.core.api.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.core.api.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the Controller service account to use
*/}}
{{- define "nhx-core.controllerServiceAccountName" -}}
{{- if .Values.core.controller.serviceAccount.create }}
{{- default (printf "%s-core-controller" (include "nemo-helix.fullname" .)) .Values.core.controller.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.core.controller.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Detect legacy controller selection flags passed through core.controller.extraArgs.
*/}}
{{- define "nhx-core.hasControllerSelectionExtraArgs" -}}
{{- $hasSelection := false -}}
{{- range .Values.core.controller.extraArgs }}
{{- $arg := toString . -}}
{{- if or (eq $arg "--controllers") (hasPrefix "--controllers=" $arg) (eq $arg "--controller-group") (hasPrefix "--controller-group=" $arg) -}}
{{- $hasSelection = true -}}
{{- end -}}
{{- end -}}
{{- if $hasSelection -}}
true
{{- end -}}
{{- end }}

{{/*
Render the `nemo services run` controller selection arg for the controller pod.
core.controller.controllers takes precedence over core.controller.controllerGroup.
*/}}
{{- define "nhx-core.controllerSelectionArgs" -}}
{{- if not (include "nhx-core.hasControllerSelectionExtraArgs" .) -}}
{{- if and (hasKey .Values.core.controller "controllers") (not (kindIs "slice" .Values.core.controller.controllers)) -}}
{{- fail "core.controller.controllers must be a list when set" -}}
{{- end -}}
{{- $controllers := .Values.core.controller.controllers | default list -}}
{{- if $controllers -}}
{{- $controllerList := join "," $controllers -}}
{{- if not ($controllerList | trim) -}}
{{- fail "core.controller.controllers must not be empty when set" -}}
{{- end -}}
- {{ printf "--controllers=%s" $controllerList | quote }}
{{- else if .Values.core.controller.controllerGroup -}}
- {{ printf "--controller-group=%s" .Values.core.controller.controllerGroup | quote }}
{{- else -}}
{{- fail "one of core.controller.controllerGroup or core.controller.controllers must be set for the controller deployment" -}}
{{- end -}}
{{- end -}}
{{- end }}

{{/*
Create the name of the Jobs service account to use (for pods created by the jobs controller)
*/}}
{{- define "nhx-core.jobsServiceAccountName" -}}
{{- if .Values.core.jobs.serviceAccount.create }}
{{- default (printf "%s-jobs" (include "nemo-helix.fullname" .)) .Values.core.jobs.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.core.jobs.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the PVC name
*/}}
{{- define "nhx-core.persistentVolumeClaim" -}}
{{- printf "%s-core-storage" (include "nemo-helix.fullname" .) }}
{{- end }}

{{/*
Define whether local files backend is enabled
*/}}
{{- define "nhx-core.localStorageEnabled" -}}
{{- if (include "nemo-helix.calculatedConfig" . | fromYaml).files -}}
{{- eq ( (include "nemo-helix.calculatedConfig" . | fromYaml).files.default_storage_config.type ) "local" -}}
{{- else -}}
false
{{- end -}}
{{- end -}}

{{/*
Create the local storage path for files
*/}}
{{- define "nhx-core.localStoragePath" -}}
{{- if (include "nemo-helix.calculatedConfig" . | fromYaml).files -}}
{{ (include "nemo-helix.calculatedConfig" . | fromYaml).files.default_storage_config.path | default "" }}
{{- end -}}
{{- end }}
