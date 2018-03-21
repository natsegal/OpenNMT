--[[ CorrectionConstraints is an implementation of the Constraints interface
  for spelling correction via insertion and deletion tags : <ins>, </ins>, <del>, </del>.
  Any other modification of the source is forbidden.
--]]
local CorrectionConstraints = torch.class('CorrectionConstraints', 'Constraints')

--[[ Constructor.

Parameters:

  * `srcTokens` - a tensor wth first dimension of batch size and second dimension of max source length, src tokens encoded with target dictionary.
  * `tags` - a table with tag token indexes : <ins>, </ins>, <del>, </del>.

--]]
function CorrectionConstraints:__init(srcTokens, tags)
  self.srcTokens = srcTokens

  self.insIdx = tags[1]
  self.insEndIdx = tags[2]
  self.delIdx = tags[3]
  self.delEndIdx = tags[4]

  self.noCorrState = 0
  self.insState = 1
  self.delState = 2
end

--[[Initilizes constraint states and constraint contents for the first beam.

Parameters:

  * `batchSize` - current batch size.

Returns:

  * `constraintIds` - a tensor of batchSize.
  * `constraintContent` - a tensor with first dimension of batchSize.

]]
function CorrectionConstraints:initConstraints(batchSize)
  local constraintIds = onmt.utils.Cuda.convert(torch.IntTensor(batchSize)):fill(self.noCorrState)
  local srcIndexes = onmt.utils.Cuda.convert(torch.IntTensor(batchSize)):fill(1)
  return constraintIds, srcIndexes
end

--[[Masks scores for forbidden tokens at each time step, based on current constraint states and constraint content.

Parameters:

  * `scores` - a tensor with first dimension of batchxbeam size and second dimension of vocabulary size.
  * `constraintIds` - a flat tensor of batchxbeam size, current constraint Ids.
  * `constraintContent` - a tensor with first dimension of batchxbeam size, current constraint content.

Returns: modifies `scores`.

]]
function CorrectionConstraints:maskScores(scores, constraintIds, constraintContent, remainingIds, beamSize) -- TODO : format parameters scores for the next token, current constraint state identifiers, current src token idx

  for c = 1, constraintIds:size(1) do

    local srcIdx = constraintContent[c]

    local batch = math.floor((c-1)/beamSize) + 1
    if #remainingIds > 0 and #remainingIds ~= self.srcTokens:size(1) then
      batch = remainingIds[batch]
    end

    local srcToken
    if srcIdx <= self.srcTokens:size(2) and self.srcTokens[batch][srcIdx] ~= 0 then
      srcToken = self.srcTokens[batch][srcIdx]
    end

    -- Not inside a correction
    -- Allow current src token, <ins>, <del>
    if constraintIds[c] == self.noCorrState then

      local srcScore
      local delScore
      local eosScore
      
      -- src tokens are not exhausted
      if srcToken then
        srcScore = scores[c][srcToken]
	delScore = scores[c][self.delIdx]
      -- src tokens are exhausted
      else
        eosScore = scores[c][onmt.Constants.EOS]
      end
      local insScore = scores[c][self.insIdx]

      scores[c]:fill(-math.huge)

      if srcToken then
        scores[c][srcToken] = srcScore
	scores[c][self.delIdx] = delScore
      else
        scores[c][onmt.Constants.EOS] = eosScore
      end
      scores[c][self.insIdx] = insScore
    end

    -- Inside an insertion
    -- Allow any token except <del>, </del>, <ins>, PAD, BOS, EOS, UNK
    if constraintIds[c] == self.insState then -- inside an insertion
      scores[c][self.delIdx] = -math.huge
      scores[c][self.delEndIdx] = -math.huge
      scores[c][self.insIdx] = -math.huge
      scores[c][onmt.Constants.PAD] = -math.huge
      scores[c][onmt.Constants.BOS] = -math.huge
      scores[c][onmt.Constants.EOS] = -math.huge
      scores[c][onmt.Constants.UNK] = -math.huge
    end

    -- Inside a deletion
    -- Allow src token, </del>
    if constraintIds[c] == self.delState then
      local srcScore
      if srcToken then
        srcScore = scores[c][srcToken]
      end
      local delEndScore = scores[c][self.delEndIdx]

      scores[c]:fill(-math.huge)
      if srcToken then
        scores[c][srcToken] = srcScore
      end
      scores[c][self.delEndIdx] = delEndScore
    end
  end
end

--[[Updates constraint states based on selected best tokens and previous constraint states and context.

Parameters:

  * `nextTokens` - a flat tensor of batchxbeam size, new beam tokens.
  * `constraintIds` - a flat tensor of batchxbeam size, previous constraint Ids.
  * `constraintContent` - a flat tensor with first dimension of batchxbeam size, previous constraint content.

Returns: modifies `constraintIds` and `constraintContent`.

]]
function CorrectionConstraints:updateConstraints(nextTokens, constraintIds, constraintContent, remainingIds, beamSize)

  for c = 1, constraintIds:size(1) do
    local srcIdx = constraintContent[c]

    local batch = math.floor((c-1)/beamSize) + 1
    if #remainingIds > 0 and #remainingIds ~= self.srcTokens:size(1) then
      batch = remainingIds[batch]
    end

    local srcToken
    if srcIdx <= self.srcTokens:size(2) and self.srcTokens[batch][srcIdx] ~= 0 then
      srcToken = self.srcTokens[batch][srcIdx]
    end

    local nextToken = nextTokens[c]
    local nextState
    local nextSrcIdx = srcIdx

    -- Not inside a correction
    if constraintIds[c] == self.noCorrState then
      if nextToken == self.insIdx then
        nextState = self.insState
      elseif nextToken == self.delIdx then
        nextState = self.delState
      elseif nextToken == srcToken then
        nextState = self.noCorrState
        nextSrcIdx = nextSrcIdx + 1
      end
    end

    -- Inside an insertion
    if constraintIds[c] == self.insState then
      if nextToken == self.insEndIdx then
        nextState = self.noCorrState
      else
        local prohibitedTokens = { [self.delIdx] = true, [self.delEndIdx] = true, [self.insIdx] = true, [onmt.Constants.PAD] = true, [onmt.Constants.BOS] = true, [onmt.Constants.EOS] = true, [onmt.Constants.UNK] = true }
        if not prohibitedTokens[nextToken] then
          nextState = self.insState
	end
      end
    end

    -- Inside a deletion
    if constraintIds[c] == self.delState then
      if nextToken == srcToken then
        nextState = self.delState
	nextSrcIdx = nextSrcIdx + 1
      elseif nextToken == self.delEndIdx then
        nextState = self.noCorrState
      end
    end

    if nextState then
      constraintIds[c] = nextState
      constraintContent[c] = nextSrcIdx
    end
  end
end

return CorrectionConstraints
