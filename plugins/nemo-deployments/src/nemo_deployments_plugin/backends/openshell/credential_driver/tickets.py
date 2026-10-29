# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export the signed registration request types used by the credential driver."""

from nemo_helix_plugin.openshell.registration import (
    MAX_BINDING_LIFETIME as MAX_BINDING_LIFETIME,
)
from nemo_helix_plugin.openshell.registration import (
    MAX_TICKET_LIFETIME as MAX_TICKET_LIFETIME,
)
from nemo_helix_plugin.openshell.registration import (
    TICKET_PREFIX as TICKET_PREFIX,
)
from nemo_helix_plugin.openshell.registration import (
    BindingScope as BindingScope,
)
from nemo_helix_plugin.openshell.registration import (
    RegistrationDenied as RegistrationDenied,
)
from nemo_helix_plugin.openshell.registration import (
    RegistrationTickets as RegistrationTickets,
)
from nemo_helix_plugin.openshell.registration import (
    TicketClaims as TicketClaims,
)
