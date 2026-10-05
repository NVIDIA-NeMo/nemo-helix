{{/*
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "nemo-helix.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "nemo-helix.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "nemo-helix.labels" -}}
helm.sh/chart: {{ include "nemo-helix.chart" . }}
{{ include "nemo-helix.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "nemo-helix.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nemo-helix.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Calculate the config from structured and unrendered base platform, with overrides
*/}}
{{- define "nemo-helix.calculatedConfig" -}}
{{ tpl (mergeOverwrite (include "nemo-helix.unstructuredConfig" . | fromYaml) .Values.platformConfig | toYaml) . }}
{{- end -}}

{{/*
Calculate the config from the unrendered base platform, before any overrides
*/}}
{{- define "nemo-helix.unstructuredConfig" -}}
{{ include (print $.Template.BasePath "/_config-render.tpl") . }}
{{- end -}}

{{/*
Determine if authentication is enabled from the calculated platform config (platformConfig.auth.enabled).
Returns "true" when auth is enabled, empty string otherwise. Use with: {{- if include "nemo-helix.authEnabled" . }}
*/}}
{{- define "nemo-helix.authEnabled" -}}
{{- $config := include "nemo-helix.calculatedConfig" . | fromYaml -}}
{{- if and $config $config.auth (eq $config.auth.enabled true) -}}
true
{{- end -}}
{{- end -}}

{{/*
Determine if the calculated platform config uses the embedded PDP provider.
Returns "true" when auth is enabled and auth.policy_decision_point_provider is "embedded".
*/}}
{{- define "nemo-helix.embeddedPdpEnabled" -}}
{{- $config := include "nemo-helix.calculatedConfig" . | fromYaml -}}
{{- if and $config $config.auth (eq $config.auth.enabled true) (eq $config.auth.policy_decision_point_provider "embedded") -}}
true
{{- end -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "nemo-helix.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create the name of the configmap to use
*/}}
{{- define "nemo-helix.platform-configmap" -}}
{{- printf "%s-config" (include "nemo-helix.fullname" .) }}
{{- end }}

{{/*
Default backend service name for ingress/HTTPRoute/OpenShift Route.
When auth and Envoy proxy are enabled, returns the Envoy service name; otherwise the API service name.
Use in values (e.g. ingress.hosts[].paths[].service) with tpl so routing points to the correct backend.
*/}}
{{- define "nemo-helix.ingressBackendService" -}}
{{- if and (include "nemo-helix.authEnabled" .) .Values.envoyProxy.enabled -}}
{{ include "nhx-envoy.servicename" . }}
{{- else -}}
{{ include "nhx-api.api-servicename" . }}
{{- end -}}
{{- end -}}

{{/*
Default backend port for ingress/HTTPRoute/OpenShift Route.
When auth and Envoy proxy are enabled, returns the Envoy service port; otherwise the API service port.
Use in values (e.g. ingress.hosts[].paths[].port) with tpl so routing points to the correct backend.
*/}}
{{- define "nemo-helix.ingressBackendPort" -}}
{{- if and (include "nemo-helix.authEnabled" .) .Values.envoyProxy.enabled -}}
{{ .Values.envoyProxy.service.port }}
{{- else -}}
{{ .Values.api.service.port }}
{{- end -}}
{{- end -}}

{{/*
Bind address for in-cluster platform runner pods.
*/}}
{{- define "nemo-helix.bindHost" -}}
{{- $config := include "nemo-helix.calculatedConfig" . | fromYaml -}}
{{- dig "service" "host" "0.0.0.0" $config -}}
{{- end -}}

{{/*
Internal API URL for pods that need to call the platform API service.
*/}}
{{- define "nemo-helix.internalBaseUrl" -}}
{{- printf "http://%s:%s" (include "nhx-api.api-servicename" .) (toString .Values.api.service.port) -}}
{{- end -}}

{{/*
Loopback API URL for the API pod itself when embedded auth must call back into the
local process instead of the cluster Service.
*/}}
{{- define "nemo-helix.apiLoopbackBaseUrl" -}}
{{- printf "http://localhost:%s" (toString .Values.api.service.port) -}}
{{- end -}}

{{/*
Validate that Envoy retires idle upstream API connections before Uvicorn closes
them. This avoids reusing a backend connection that the API already dropped.
*/}}
{{- define "nemo-helix.validateEnvoyKeepAliveTimeouts" -}}
{{- if and .Values.api.enabled (include "nemo-helix.authEnabled" .) .Values.envoyProxy.enabled -}}
{{- $apiKeepAliveSeconds := .Values.api.server.keepAliveTimeoutSeconds | int -}}
{{- if lt $apiKeepAliveSeconds 1 -}}
{{- fail "api.server.keepAliveTimeoutSeconds must be greater than 0" -}}
{{- end -}}
{{- $upstreamIdle := .Values.envoyProxy.timeouts.upstreamIdle | toString -}}
{{- if not (regexMatch "^[1-9][0-9]*s$" $upstreamIdle) -}}
{{- fail "envoyProxy.timeouts.upstreamIdle must be a positive whole-second duration like \"4s\"" -}}
{{- end -}}
{{- $upstreamIdleSeconds := trimSuffix "s" $upstreamIdle | int -}}
{{- if ge $upstreamIdleSeconds $apiKeepAliveSeconds -}}
{{- fail (printf "envoyProxy.timeouts.upstreamIdle (%s) must be less than api.server.keepAliveTimeoutSeconds (%ds)" $upstreamIdle $apiKeepAliveSeconds) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Pod annotations
*/}}
{{- define "nemo-helix.podAnnotations" -}}
checksum/config: {{ include (print $.Template.BasePath "/platform-configmap.yaml") . | sha256sum }}
{{- end -}}

{{/*
Name of the Secret containing API environment values or the secrets service
default encryption key. Existing default-encryption-key Secrets are mounted with
secretKeyRef instead of envFrom.
*/}}
{{- define "nemo-helix.apiEnvSecretName" -}}
{{- .Values.envFromSecret | default (include "nemo-helix.defaultEncryptionKeyExistingSecretName" .) | default (printf "%s-api-env" (include "nemo-helix.fullname" .)) -}}
{{- end -}}

{{/*
Environment variable name used by the secrets service secret_key provider.
*/}}
{{- define "nemo-helix.defaultEncryptionKeyEnvName" -}}
NHX_SECRETS_DEFAULT_ENCRYPTION_KEY
{{- end -}}

{{/*
Existing Secret name for the secrets service default encryption key.
*/}}
{{- define "nemo-helix.defaultEncryptionKeyExistingSecretName" -}}
{{- .Values.secrets.defaultEncryptionKey.existingSecret.name | default "" -}}
{{- end -}}

{{/*
Secret key that contains the secrets service default encryption key.
*/}}
{{- define "nemo-helix.defaultEncryptionKeySecretKey" -}}
{{- if include "nemo-helix.defaultEncryptionKeyExistingSecretName" . -}}
{{- .Values.secrets.defaultEncryptionKey.existingSecret.key | default (include "nemo-helix.defaultEncryptionKeyEnvName" .) -}}
{{- else -}}
{{- include "nemo-helix.defaultEncryptionKeyEnvName" . -}}
{{- end -}}
{{- end -}}

{{/*
Whether the chart should generate the API env Secret through a pre-install hook.
Generation is install-only. On upgrade, a missing generated key is unrecoverable
without restoring the original key or rotating/re-encrypting secrets through the
supported admin flow, so the chart must not generate a replacement.
*/}}
{{- define "nemo-helix.generateDefaultEncryptionKey" -}}
{{- if and .Release.IsInstall (not .Values.envFromSecret) (not (include "nemo-helix.defaultEncryptionKeyExistingSecretName" .)) (not .Values.secrets.defaultEncryptionKey.value) .Values.secrets.defaultEncryptionKey.generated.enabled -}}
true
{{- end -}}
{{- end -}}

{{/*
Whether an upgrade should require the generated API env Secret to already exist.
*/}}
{{- define "nemo-helix.requireExistingGeneratedDefaultEncryptionKey" -}}
{{- if and .Release.IsUpgrade (not .Values.envFromSecret) (not (include "nemo-helix.defaultEncryptionKeyExistingSecretName" .)) (not .Values.secrets.defaultEncryptionKey.value) .Values.secrets.defaultEncryptionKey.generated.enabled -}}
true
{{- end -}}
{{- end -}}

{{/*
Name shared by the key generation hook RBAC and Job resources.
*/}}
{{- define "nemo-helix.defaultEncryptionKeyGeneratorName" -}}
{{- printf "%s-api-env-keygen" (include "nemo-helix.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
ServiceAccount name for the key generation hook.
*/}}
{{- define "nemo-helix.defaultEncryptionKeyGeneratorServiceAccountName" -}}
{{- if .Values.secrets.defaultEncryptionKey.generated.serviceAccount.create -}}
{{- default (include "nemo-helix.defaultEncryptionKeyGeneratorName" .) .Values.secrets.defaultEncryptionKey.generated.serviceAccount.name -}}
{{- else -}}
{{- required "secrets.defaultEncryptionKey.generated.serviceAccount.name is required when secrets.defaultEncryptionKey.generated.serviceAccount.create is false" .Values.secrets.defaultEncryptionKey.generated.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Image Pull Secrets
*/}}
{{- define "nemo-common.imagepullsecrets" -}}
{{- with .Values.imagePullSecrets -}}
{{- toYaml . -}}
{{- end -}}
{{- end }}

{{/*
JSON list of image pull secret names for scripts that create pods.
*/}}
{{- define "nemo-common.imagepullsecretnames" -}}
{{- $names := list -}}
{{- range .Values.imagePullSecrets -}}
{{- $names = append $names .name -}}
{{- end -}}
{{- toJson $names -}}
{{- end }}

{{/*
Embedded PostgreSQL full name (service and secret name when postgresql.enabled).
*/}}
{{- define "nemo-common.postgresql.fullname" -}}
{{- printf "%s-postgres" (include "nemo-helix.fullname" . | trunc 54 | trimSuffix "-") -}}
{{- end -}}

{{/*
Name of the service account to use for the embedded PostgreSQL pod.
*/}}
{{- define "nemo-common.postgresql.serviceAccountName" -}}
{{- if .Values.postgresql.serviceAccount.create -}}
{{- default (printf "%s-postgres" (include "nemo-helix.fullname" .)) .Values.postgresql.serviceAccount.name }}
{{- else -}}
{{- default "default" .Values.postgresql.serviceAccount.name }}
{{- end -}}
{{- end -}}

{{/*
PostgreSQL Hostname
*/}}
{{- define "nemo-common.postgresql.host" -}}
{{- if .Values.postgresql.enabled -}}
{{ include "nemo-common.postgresql.fullname" . }}
{{- else -}}
{{ .Values.externalDatabase.host }}
{{- end -}}
{{- end -}}

{{/*
nemo-common.database.port chooses between externalDatabase and the embedded postgresql port
*/}}
{{- define "nemo-common.postgresql.port" -}}
{{- if .Values.postgresql.enabled -}}
{{- printf "%d" (.Values.postgresql.service.port | int) -}}
{{- else -}}
{{- printf "%d" (.Values.externalDatabase.port | int) -}}
{{- end -}}
{{- end -}}

{{/*
nemo-common.database.user chooses between externalDatabase and the embedded postgresql user values
*/}}
{{- define "nemo-common.postgresql.user" -}}
{{- if .Values.postgresql.enabled -}}
{{- print .Values.postgresql.auth.username -}}
{{- else -}}
{{- print .Values.externalDatabase.user -}}
{{- end -}}
{{- end -}}

{{/*
nemo-common.database.name chooses between externalDatabase and the embedded postgresql db name values
*/}}
{{- define "nemo-common.postgresql.name" -}}
{{- if .Values.postgresql.enabled -}}
{{- print .Values.postgresql.auth.database -}}
{{- else -}}
{{- print .Values.externalDatabase.database -}}
{{- end -}}
{{- end -}}

{{/*
nemo-common.database.secret-name chooses between externalDatabase and the embedded postgresql existing secret values
*/}}
{{- define "nemo-common.postgresql.secret-name" -}}
{{- if .Values.postgresql.enabled -}}
{{- if .Values.postgresql.auth.existingSecret -}}
{{- print .Values.postgresql.auth.existingSecret -}}
{{- else -}}
{{ include "nemo-common.postgresql.fullname" . }}
{{- end -}}
{{- else if .Values.externalDatabase.existingSecret -}}
{{- print .Values.externalDatabase.existingSecret -}}
{{- end -}}
{{- end -}}

{{/*
nemo-common.database.password-key chooses between externalDatabase and the embedded postgresql existing secret key values
*/}}
{{- define "nemo-common.postgresql.password-key" -}}
{{- if or .Values.postgresql.enabled (not .Values.externalDatabase.existingSecret) -}}
{{- print "password" -}}
{{- else -}}
{{- print .Values.externalDatabase.existingSecretPasswordKey -}}
{{- end -}}
{{- end -}}

{{/*
nemo-common.database.password generates a POSTGRES_DB_PASSWORD environment value if a full URI isn't used
*/}}
{{- define "nemo-common.postgresql.password" -}}
{{- if not (and .Values.externalDatabase.uriSecret .Values.externalDatabase.uriSecret.name .Values.externalDatabase.uriSecret.key) }}
- name: POSTGRES_DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "nemo-common.postgresql.secret-name" .}}
      key: {{ include "nemo-common.postgresql.password-key" .}}
{{- end }}
{{- end -}}

{{/*
Embedded ClickHouse full name (service and generated secret name).
*/}}
{{- define "nemo-common.clickhouse.fullname" -}}
{{- printf "%s-clickhouse" (include "nemo-helix.fullname" . | trunc 51 | trimSuffix "-") -}}
{{- end -}}

{{/*
Name of the service account to use for the embedded ClickHouse pod.
*/}}
{{- define "nemo-common.clickhouse.serviceAccountName" -}}
{{- if .Values.clickhouse.serviceAccount.create -}}
{{- default (include "nemo-common.clickhouse.fullname" .) .Values.clickhouse.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.clickhouse.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Whether the embedded ClickHouse should be rendered.
*/}}
{{- define "nemo-common.clickhouse.enabled" -}}
{{- if .Values.clickhouse.enabled -}}
true
{{- end -}}
{{- end -}}

{{/*
ClickHouse HTTP URL used by Intake.
*/}}
{{- define "nemo-common.clickhouse.url" -}}
{{- if .Values.clickhouse.enabled -}}
{{- printf "http://%s:%d" (include "nemo-common.clickhouse.fullname" .) (.Values.clickhouse.service.httpPort | int) -}}
{{- else -}}
{{- $host := required "externalClickhouse.host is required when clickhouse.enabled=false" .Values.externalClickhouse.host -}}
{{- $scheme := ternary "https" "http" .Values.externalClickhouse.secure -}}
{{- printf "%s://%s:%d" $scheme $host (.Values.externalClickhouse.port | int) -}}
{{- end -}}
{{- end -}}

{{/*
ClickHouse username used by Intake.
*/}}
{{- define "nemo-common.clickhouse.user" -}}
{{- if .Values.clickhouse.enabled -}}
{{- .Values.clickhouse.auth.username -}}
{{- else -}}
{{- .Values.externalClickhouse.user -}}
{{- end -}}
{{- end -}}

{{/*
ClickHouse database used by Intake.
*/}}
{{- define "nemo-common.clickhouse.database" -}}
{{- if .Values.clickhouse.enabled -}}
{{- .Values.clickhouse.auth.database -}}
{{- else -}}
{{- .Values.externalClickhouse.database -}}
{{- end -}}
{{- end -}}

{{/*
Secret containing the ClickHouse password used by Intake.
*/}}
{{- define "nemo-common.clickhouse.secretName" -}}
{{- if .Values.clickhouse.enabled -}}
{{- default (include "nemo-common.clickhouse.fullname" .) .Values.clickhouse.auth.existingSecret -}}
{{- else -}}
{{- required "externalClickhouse.existingSecret is required when clickhouse.enabled=false" .Values.externalClickhouse.existingSecret -}}
{{- end -}}
{{- end -}}

{{/*
Key containing the ClickHouse password used by Intake.
*/}}
{{- define "nemo-common.clickhouse.passwordKey" -}}
{{- if .Values.clickhouse.enabled -}}
{{- if .Values.clickhouse.auth.existingSecret -}}
{{- .Values.clickhouse.auth.existingSecretPasswordKey | default "password" -}}
{{- else -}}
password
{{- end -}}
{{- else -}}
{{- required "externalClickhouse.existingSecretPasswordKey is required when clickhouse.enabled=false" .Values.externalClickhouse.existingSecretPasswordKey -}}
{{- end -}}
{{- end -}}

{{/*
Checksum of the ClickHouse password used by Intake. Prefer an explicitly
configured password for a chart-managed Secret, then the live Secret value, so
credential changes roll both API and ClickHouse pods together. During first
install with a generated password, use a deterministic fallback until the
Secret exists.
*/}}
{{- define "nemo-common.clickhouse.credentialsChecksum" -}}
{{- $_ := include "nemo-common.clickhouse.url" . -}}
{{- $secretName := include "nemo-common.clickhouse.secretName" . -}}
{{- $passwordKey := include "nemo-common.clickhouse.passwordKey" . -}}
{{- $secret := lookup "v1" "Secret" .Release.Namespace $secretName -}}
{{- $passwordData := get ($secret.data | default dict) $passwordKey | default "" -}}
{{- if and .Values.clickhouse.enabled (not .Values.clickhouse.auth.existingSecret) .Values.clickhouse.auth.password -}}
{{- .Values.clickhouse.auth.password | toString | b64enc | sha256sum -}}
{{- else if $passwordData -}}
{{- $passwordData | sha256sum -}}
{{- else -}}
{{- printf "%s:%s" $secretName $passwordKey | sha256sum -}}
{{- end -}}
{{- end -}}

{{/*
nemo-common.otel-env generates an env var array from the top-level telemetry configuration.
Follows the specification at https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/

Usage:
  {{ include "nemo-common.otel-env" (dict "root" $ "local" .Values) }}
*/}}
{{- define "nemo-common.otel-env" -}}
{{- $root := .root -}}
{{- $local := .local -}}
{{- $globalOtel := $root.Values.telemetry | default dict -}}
{{- $localOtel := $local.telemetry | default dict -}}

{{- $merged := dict -}}
{{- range $k, $v := $globalOtel }}
  {{- $_ := set $merged $k $v }}
{{- end }}
{{- range $k, $v := $localOtel }}
  {{- $_ := set $merged $k $v }}
{{- end }}

{{- range $key, $val := $merged }}
{{- if not (kindIs "invalid" $val) }}
- name: {{ $key }}
  value: {{ $val | quote }}
{{- end }}
{{- end }}
{{- end -}}

{{/*
nemo-common.podSecurityContext merges global podSecurityContext with component-specific podSecurityContext.
Component values override global values.

Usage:
  {{ include "nemo-common.podSecurityContext" (dict "global" .Values.podSecurityContext "local" .Values.api.podSecurityContext) }}
*/}}
{{- define "nemo-common.podSecurityContext" -}}
{{- $global := .global | default dict -}}
{{- $local := .local | default dict -}}
{{- $merged := mergeOverwrite (deepCopy $global) $local -}}
{{- if $merged }}
{{- toYaml $merged }}
{{- end }}
{{- end -}}

{{/*
nemo-common.securityContext merges global securityContext with component-specific securityContext.
Component values override global values.

Usage:
  {{ include "nemo-common.securityContext" (dict "global" .Values.securityContext "local" .Values.api.securityContext) }}
*/}}
{{- define "nemo-common.securityContext" -}}
{{- $global := .global | default dict -}}
{{- $local := .local | default dict -}}
{{- $merged := mergeOverwrite (deepCopy $global) $local -}}
{{- if $merged }}
{{- toYaml $merged }}
{{- end }}
{{- end -}}

{{/*
Determine if multi-node networking is enabled for any cloud provider.
Returns "true" if any cloud provider networking is enabled, empty string otherwise.

Usage:
  {{- if include "nemo-helix.multinodeNetworkingEnabled" . }}
*/}}
{{- define "nemo-helix.multinodeNetworkingEnabled" -}}
{{- if or .Values.multinodeNetworking.aws.enabled .Values.multinodeNetworking.azure.enabled .Values.multinodeNetworking.gcp.enabled .Values.multinodeNetworking.oci.enabled -}}
true
{{- end -}}
{{- end -}}

{{/*
nemo-helix.env generates an env var array out of a dict to allow better
interleaving, easier use and default settings. It will still work if a you use
an array to render directly, but it is not recommended. It is available across
all pods.
*/}}
{{- define "nemo-helix.env" -}}
{{- if and .Values.env (kindIs "slice" .Values.env) -}}
{{- toYaml .Values.env -}}
{{- else if and .Values.env (kindIs "map" .Values.env) -}}
{{- range $k, $v := .Values.env }}
- name: {{ $k }}
  {{- if kindIs "map" $v }}
  valueFrom:
    {{ toYaml $v.valueFrom | nindent 4 | trim }}
  {{- else }}
  value: {{ $v | quote }}
  {{- end }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
nemo-helix.api.env generates an env var array out of a dict to allow better
interleaving, easier use and default settings. It will still work if a you use
an array to render directly, but it is not recommended. It is available ONLY to the api pod.
*/}}
{{- define "nemo-helix.api.env" -}}
{{- if and .Values.api.env (kindIs "slice" .Values.api.env) -}}
{{- toYaml .Values.api.env -}}
{{- else if and .Values.api.env (kindIs "map" .Values.api.env) -}}
{{- range $k, $v := .Values.api.env }}
- name: {{ $k }}
  {{- if kindIs "map" $v }}
  valueFrom:
    {{ toYaml $v.valueFrom | nindent 4 | trim }}
  {{- else }}
  value: {{ $v | quote }}
  {{- end }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
nemo-helix.controller.env generates an env var array out of a dict to allow better
interleaving, easier use and default settings. It will still work if a you use
an array to render directly, but it is not recommended. It is available ONLY to the controller pod.
*/}}
{{- define "nemo-helix.controller.env" -}}
{{- if and .Values.core.controller.env (kindIs "slice" .Values.core.controller.env) -}}
{{- toYaml .Values.core.controller.env -}}
{{- else if and .Values.core.controller.env (kindIs "map" .Values.core.controller.env) -}}
{{- range $k, $v := .Values.core.controller.env }}
- name: {{ $k }}
  {{- if kindIs "map" $v }}
  valueFrom:
    {{ toYaml $v.valueFrom | nindent 4 | trim }}
  {{- else }}
  value: {{ $v | quote }}
  {{- end }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
nemo-helix.envoyProxy.env generates an env var array from .Values.envoyProxy.env (map of
NAME: value or NAME: valueFrom: {object}). Same format as nemo-helix.api.env.
*/}}
{{- define "nemo-helix.envoyProxy.env" -}}
{{- if and .Values.envoyProxy.env (kindIs "slice" .Values.envoyProxy.env) -}}
{{- toYaml .Values.envoyProxy.env -}}
{{- else if and .Values.envoyProxy.env (kindIs "map" .Values.envoyProxy.env) -}}
{{- range $k, $v := .Values.envoyProxy.env }}
- name: {{ $k }}
  {{- if kindIs "map" $v }}
  valueFrom:
    {{ toYaml $v.valueFrom | nindent 4 | trim }}
  {{- else }}
  value: {{ $v | quote }}
  {{- end }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
Create the name of the models files auth secret (HF_TOKEN for Files service pull-through).
*/}}
{{- define "nemo-helix.modelsFilesAuthSecretName" -}}
{{- printf "%s-models-files-token" (include "nemo-helix.fullname" .) }}
{{- end }}
