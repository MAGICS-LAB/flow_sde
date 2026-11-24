# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Import the flow-policy PPO implementations for easy access.
# - PPOFlow: Original ReinFlow implementation (from ppoflow.py)
# - FlowSDE: FMPO implementation with tunable noise schedule (from ppoflow_fmppo.py).
#            FMPO-ChainPPO is enabled inside FlowSDE with use_chain_ppo_loss=True.
#
# Usage in config files:
#   ReinFlow:       model._target_: model.flow.ft_ppo.ppoflow.PPOFlow
#   FlowSDE:        model._target_: model.flow.ft_ppo.ppoflow_fmppo.FlowSDE
#   FMPO-ChainPPO:  model._target_: model.flow.ft_ppo.ppoflow_fmppo.FlowSDE
#                   model.use_chain_ppo_loss=true

from model.flow.ft_ppo.ppoflow import PPOFlow
from model.flow.ft_ppo.ppoflow_fmppo import FlowSDE, PPOFlow as PPOFlowFMPO

__all__ = ['PPOFlow', 'PPOFlowFMPO', 'FlowSDE']
