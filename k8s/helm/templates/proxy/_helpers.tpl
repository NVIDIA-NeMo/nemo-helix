{{/*
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{/*
Create a named Envoy service name which can be included from parent chart
*/}}
{{- define "nhx-envoy.servicename" }}
{{- printf "%s-envoy" ( include "nemo-helix.fullname" . | trunc 57 ) }}
{{- end }}

{{/*
Create the Envoy image reference.
*/}}
{{- define "nhx-envoy.image" -}}
{{- if .Values.envoyProxy.image.digest -}}
{{ printf "%s@%s" .Values.envoyProxy.image.repository .Values.envoyProxy.image.digest }}
{{- else -}}
{{ printf "%s:%s" .Values.envoyProxy.image.repository .Values.envoyProxy.image.tag }}
{{- end -}}
{{- end }}

{{/*
Labels for Envoy proxy resources (component + platform labels).
*/}}
{{- define "nhx-envoy.labels" -}}
app.kubernetes.io/component: nhx-envoy
{{ include "nemo-helix.labels" . }}
{{- end }}

{{/*
Create the name of the Envoy service account to use
*/}}
{{- define "nhx-envoy.serviceAccountName" -}}
{{- if .Values.envoyProxy.serviceAccount.create }}
{{- default (printf "%s-envoy" (include "nemo-helix.fullname" .)) .Values.envoyProxy.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.envoyProxy.serviceAccount.name }}
{{- end }}
{{- end }}
