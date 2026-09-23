{{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{- define "nemo-helix-zitadel.namespace" -}}
{{- .Release.Namespace -}}
{{- end -}}

{{- define "nemo-helix-zitadel.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | quote }}
app.kubernetes.io/name: {{ .Chart.Name | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service | quote }}
{{- end -}}

{{- define "nemo-helix-zitadel.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
{{- end -}}

{{- define "nemo-helix-zitadel.serviceNamespacedHost" -}}
{{- $namespace := .namespace | default .root.Release.Namespace -}}
{{- printf "%s.%s" .serviceName $namespace -}}
{{- end -}}

{{- define "nemo-helix-zitadel.serviceFqdn" -}}
{{- $clusterDomain := .clusterDomain | default "cluster.local" -}}
{{- printf "%s.svc.%s" (include "nemo-helix-zitadel.serviceNamespacedHost" .) $clusterDomain -}}
{{- end -}}

{{- define "nemo-helix-zitadel.serviceUrl" -}}
{{- $host := include "nemo-helix-zitadel.serviceFqdn" . -}}
{{- if hasKey . "port" -}}
{{- printf "%s://%s:%s" .scheme $host (toString .port) -}}
{{- else -}}
{{- printf "%s://%s" .scheme $host -}}
{{- end -}}
{{- end -}}

{{- define "nemo-helix-zitadel.publicGatewayUrl" -}}
{{- $nemoHelixValues := index .Values "nemo-helix" | default dict -}}
{{- $gateway := .Values.zitadelPublicGateway | default (index $nemoHelixValues "zitadelPublicGateway") -}}
{{- $gateway = required "nemo-helix.zitadelPublicGateway is required" $gateway -}}
{{- $scheme := required "nemo-helix.zitadelPublicGateway.scheme is required" (index $gateway "scheme") -}}
{{- $host := required "nemo-helix.zitadelPublicGateway.host is required" (index $gateway "host") -}}
{{- $port := required "nemo-helix.zitadelPublicGateway.port is required" (index $gateway "port") -}}
{{- printf "%s://%s:%s" $scheme $host (toString $port) -}}
{{- end -}}

{{- define "nemo-helix-zitadel.serviceDnsNames" -}}
{{- $namespacedHost := include "nemo-helix-zitadel.serviceNamespacedHost" . -}}
names:
  - {{ .serviceName | quote }}
  - {{ $namespacedHost | quote }}
  - {{ printf "%s.svc" $namespacedHost | quote }}
  - {{ include "nemo-helix-zitadel.serviceFqdn" . | quote }}
{{- end -}}

{{- define "nemo-helix-zitadel.existingSecretData" -}}
{{- $existingSecret := lookup "v1" "Secret" .root.Release.Namespace .secretName -}}
{{- if and $existingSecret $existingSecret.data -}}
{{- $existingSecret.data | toJson -}}
{{- else -}}
{{- dict | toJson -}}
{{- end -}}
{{- end -}}

{{- define "nemo-helix-zitadel.secretValue" -}}
{{- $existingData := include "nemo-helix-zitadel.existingSecretData" (dict "root" .root "secretName" .secretName) | fromJson -}}
{{- if hasKey $existingData .key -}}
{{- index $existingData .key | b64dec -}}
{{- else -}}
{{- .generated -}}
{{- end -}}
{{- end -}}

{{- define "nemo-helix-zitadel.workloadTokenSigningKey.secretName" -}}
{{- required "workloadTokenSigningKey.secretName is required" .Values.workloadTokenSigningKey.secretName -}}
{{- end -}}

{{- define "nemo-helix-zitadel.workloadTokenSigningKey.key" -}}
{{- required "workloadTokenSigningKey.key is required" .Values.workloadTokenSigningKey.key -}}
{{- end -}}

{{- define "nemo-helix-zitadel.workloadTokenSigningKey.privateKeyPem" -}}
{{- $secretName := include "nemo-helix-zitadel.workloadTokenSigningKey.secretName" . -}}
{{- $secretKey := include "nemo-helix-zitadel.workloadTokenSigningKey.key" . -}}
{{- $privateKeyPem := .Values.workloadTokenSigningKey.privateKeyPem | default "" -}}
{{- $existingData := include "nemo-helix-zitadel.existingSecretData" (dict "root" . "secretName" $secretName) | fromJson -}}
{{- if $privateKeyPem -}}
{{- $privateKeyPem -}}
{{- else if hasKey $existingData $secretKey -}}
{{- index $existingData $secretKey | b64dec -}}
{{- else -}}
{{- genPrivateKey "rsa" -}}
{{- end -}}
{{- end -}}
