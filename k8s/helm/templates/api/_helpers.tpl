{{/*
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{/*
Image Definition Parsing
Favor not using a separate registry because it is confusing, but support it.
*/}}
{{- define "nhx-api.image" -}}
{{- if .Values.api.image.registry -}}
{{ .Values.api.image.registry }}/{{ .Values.api.image.repository }}:{{ default .Chart.AppVersion .Values.api.image.tag }}
{{- else -}}
{{ .Values.api.image.repository }}:{{ default .Chart.AppVersion .Values.api.image.tag }}
{{- end }}
{{- end }}

{{/*
Create a named api service name which can be included from parent chart
*/}}
{{- define "nhx-api.api-servicename" }}
{{- printf "%s-api" ( include "nemo-helix.fullname" . | trunc 59 ) }}
{{- end }}

{{/*
Create the name of the API service account to use
*/}}
{{- define "nhx-api.apiServiceAccountName" -}}
{{- if .Values.api.serviceAccount.create }}
{{- default (printf "%s-api" (include "nemo-helix.fullname" .)) .Values.api.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.api.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Detect legacy service selection flags passed through api.extraArgs.
*/}}
{{- define "nhx-api.hasServiceSelectionExtraArgs" -}}
{{- $hasSelection := false -}}
{{- range .Values.api.extraArgs }}
{{- $arg := toString . -}}
{{- if or (eq $arg "--services") (hasPrefix "--services=" $arg) (eq $arg "--service-group") (hasPrefix "--service-group=" $arg) -}}
{{- $hasSelection = true -}}
{{- end -}}
{{- end -}}
{{- if $hasSelection -}}
true
{{- end -}}
{{- end }}

{{/*
Render the `nemo services run` service selection arg for the API pod.
api.services takes precedence over api.serviceGroup so explicit services can be
set without also clearing the default service group value.
*/}}
{{- define "nhx-api.serviceSelectionArgs" -}}
{{- if not (include "nhx-api.hasServiceSelectionExtraArgs" .) -}}
{{- if and (hasKey .Values.api "services") (not (kindIs "slice" .Values.api.services)) -}}
{{- fail "api.services must be a list when set" -}}
{{- end -}}
{{- $services := .Values.api.services | default list -}}
{{- if $services -}}
{{- $serviceList := join "," $services -}}
{{- if not ($serviceList | trim) -}}
{{- fail "api.services must not be empty when set" -}}
{{- end -}}
- {{ printf "--services=%s" $serviceList | quote }}
{{- else if .Values.api.serviceGroup -}}
- {{ printf "--service-group=%s" .Values.api.serviceGroup | quote }}
{{- else -}}
{{- fail "one of api.serviceGroup or api.services must be set for the API deployment" -}}
{{- end -}}
{{- end -}}
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
