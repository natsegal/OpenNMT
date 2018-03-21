--[[ Class template for decoding constraints.

  Each implementation should specify how to:

   * initialize constraint states for current batch
   * mask scores for forbidden tokens for each hypothesis, based on its current constraint state
   * update constraint states based on new tokens and previous constraint states

--]]

local Constraints = torch.class('Constraints')

--[[Initilizes constraint states and constraint contents for the first beam.

Parameters:

  * `batchSize` - current batch size.

Returns:

  * `constraintIds` - a tensor of batchSize.
  * `constraintContent` - a tensor with first dimension of batchSize.

]]
function Constraints:initConstraints(batchSize)
end

--[[Masks scores for forbidden tokens at each time step, based on current constraint states and constraint content.

Parameters:

  * `scores` - a tensor with first dimension of batchxbeam size and second dimension of vocabulary size.
  * `constraintIds` - a flat tensor of batchxbeam size, current constraint Ids.
  * `constraintContent` - a tensor with first dimension of batchxbeam size, current constraint content.

Returns: modifies `scores`.

]]
function Constraints:maskScores(scores, constraintIds, constraintContent)
end

--[[Updates constraint states based on selected best tokens and previous constraint states and context.

Parameters:

  * `nextTokens` - a flat tensor of batchxbeam size, new beam tokens.
  * `constraintIds` - a flat tensor of batchxbeam size, previous constraint Ids.
  * `constraintContent` - a flat tensor with first dimension of batchxbeam size, previous constraint content.

Returns: modifies `constraintIds` and `constraintContent`.

]]
function Constraints:updateConstraints(nextTokens, constraintIds, constraintContent)
end

return Constraints