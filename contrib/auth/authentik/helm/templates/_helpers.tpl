{{/*
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{- define "nemo-helix-authentik.namespace" -}}
{{- .Release.Namespace -}}
{{- end -}}

{{- define "nemo-helix-authentik.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | quote }}
app.kubernetes.io/name: {{ .Chart.Name | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service | quote }}
{{- end -}}

{{- define "nemo-helix-authentik.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
{{- end -}}

{{- define "nemo-helix-authentik.sharedPostgresql.selectorLabels" -}}
{{ include "nemo-helix-authentik.selectorLabels" . }}
app.kubernetes.io/component: shared-postgresql
{{- end -}}

{{- define "nemo-helix-authentik.sharedPostgresql.serviceAccountName" -}}
{{- if .Values.sharedPostgresql.serviceAccount.create -}}
{{- default (printf "%s-postgres" .Values.sharedPostgresql.serviceName) .Values.sharedPostgresql.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.sharedPostgresql.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "nemo-helix-authentik.serviceNamespacedHost" -}}
{{- $namespace := .namespace | default .root.Release.Namespace -}}
{{- printf "%s.%s" .serviceName $namespace -}}
{{- end -}}

{{- define "nemo-helix-authentik.serviceFqdn" -}}
{{- $clusterDomain := .clusterDomain | default "cluster.local" -}}
{{- printf "%s.svc.%s" (include "nemo-helix-authentik.serviceNamespacedHost" .) $clusterDomain -}}
{{- end -}}

{{- define "nemo-helix-authentik.serviceUrl" -}}
{{- $host := include "nemo-helix-authentik.serviceFqdn" . -}}
{{- if hasKey . "port" -}}
{{- printf "%s://%s:%s" .scheme $host (toString .port) -}}
{{- else -}}
{{- printf "%s://%s" .scheme $host -}}
{{- end -}}
{{- end -}}

{{- define "nemo-helix-authentik.publicGatewayUrl" -}}
{{- $gateway := required "nemo-helix.authentikPublicGateway is required" .Values.authentikPublicGateway -}}
{{- $scheme := required "nemo-helix.authentikPublicGateway.scheme is required" (index $gateway "scheme") -}}
{{- $host := required "nemo-helix.authentikPublicGateway.host is required" (index $gateway "host") -}}
{{- $port := required "nemo-helix.authentikPublicGateway.port is required" (index $gateway "port") -}}
{{- printf "%s://%s:%s" $scheme $host (toString $port) -}}
{{- end -}}

{{- define "nemo-helix-authentik.serviceDnsNames" -}}
{{- $namespacedHost := include "nemo-helix-authentik.serviceNamespacedHost" . -}}
names:
  - {{ .serviceName | quote }}
  - {{ $namespacedHost | quote }}
  - {{ printf "%s.svc" $namespacedHost | quote }}
  - {{ include "nemo-helix-authentik.serviceFqdn" . | quote }}
{{- end -}}
