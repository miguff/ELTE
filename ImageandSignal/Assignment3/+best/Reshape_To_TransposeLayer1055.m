classdef Reshape_To_TransposeLayer1055 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    
    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    
    properties (Learnable)
        onnx__Split_418
    end
    
    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end
    
    methods
        function this = Reshape_To_TransposeLayer1055(name, onnxParams)
            this.Name = name;
            this.NumInputs = 3;
            this.NumOutputs = 3;
            this.OutputNames = {'x_model_22_dfl_Trans', 'x_model_22_Sigmoid_o', 'x_model_22_Sigmoid_oNumDims'};
            this.ONNXParams = onnxParams;
            this.onnx__Split_418 = onnxParams.Learnables.onnx__Split_418;
        end
        
        function [x_model_22_dfl_Trans, x_model_22_Sigmoid_o, x_model_22_Sigmoid_oNumDims] = predict(this, x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_)
            if isdlarray(x_model_22_Concat_ou)
                x_model_22_Concat_ou = stripdims(x_model_22_Concat_ou);
            end
            if isdlarray(x_model_22_Concat_1_)
                x_model_22_Concat_1_ = stripdims(x_model_22_Concat_1_);
            end
            if isdlarray(x_model_22_Concat_2_)
                x_model_22_Concat_2_ = stripdims(x_model_22_Concat_2_);
            end
            x_model_22_Concat_ouNumDims = 4;
            x_model_22_Concat_1_NumDims = 4;
            x_model_22_Concat_2_NumDims = 4;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.onnx__Split_418 = this.onnx__Split_418;
            [x_model_22_dfl_Trans, x_model_22_Sigmoid_o, x_model_22_dfl_TransNumDims, x_model_22_Sigmoid_oNumDims] = Reshape_To_TransposeFcn(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, x_model_22_Concat_ouNumDims, x_model_22_Concat_1_NumDims, x_model_22_Concat_2_NumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[4 3 1 2], [4 3 1 2], [4 3 1 2], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[3 4 2 1], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A), {x_model_22_dfl_Trans, x_model_22_Sigmoid_o}))
                fprintf('Runtime error in network. The custom layer ''%s'' output a non-numeric value.\n', 'Reshape_To_TransposeLayer1055');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Reshape_To_TransposeLayer1055'));
            end
            x_model_22_dfl_Trans = dlarray(single(x_model_22_dfl_Trans), 'SSCB');
            x_model_22_Sigmoid_o = dlarray(single(x_model_22_Sigmoid_o), repmat('U', 1, max(2, x_model_22_Sigmoid_oNumDims)));
            x_model_22_Sigmoid_oNumDims = dlarray(ones(1,x_model_22_Sigmoid_oNumDims,'like',x_model_22_dfl_Trans), 'UU');
            if ~coder.target('MATLAB')
                x_model_22_dfl_Trans = extractdata(x_model_22_dfl_Trans);
                x_model_22_Sigmoid_o = extractdata(x_model_22_Sigmoid_o);
                x_model_22_Sigmoid_oNumDims = extractdata(x_model_22_Sigmoid_oNumDims);
            end
        end
        
        function [x_model_22_dfl_Trans, x_model_22_Sigmoid_o, x_model_22_Sigmoid_oNumDims] = forward(this, x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_)
            if isdlarray(x_model_22_Concat_ou)
                x_model_22_Concat_ou = stripdims(x_model_22_Concat_ou);
            end
            if isdlarray(x_model_22_Concat_1_)
                x_model_22_Concat_1_ = stripdims(x_model_22_Concat_1_);
            end
            if isdlarray(x_model_22_Concat_2_)
                x_model_22_Concat_2_ = stripdims(x_model_22_Concat_2_);
            end
            x_model_22_Concat_ouNumDims = 4;
            x_model_22_Concat_1_NumDims = 4;
            x_model_22_Concat_2_NumDims = 4;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.onnx__Split_418 = this.onnx__Split_418;
            [x_model_22_dfl_Trans, x_model_22_Sigmoid_o, x_model_22_dfl_TransNumDims, x_model_22_Sigmoid_oNumDims] = Reshape_To_TransposeFcn(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, x_model_22_Concat_ouNumDims, x_model_22_Concat_1_NumDims, x_model_22_Concat_2_NumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[4 3 1 2], [4 3 1 2], [4 3 1 2], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[3 4 2 1], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A), {x_model_22_dfl_Trans, x_model_22_Sigmoid_o}))
                fprintf('Runtime error in network. The custom layer ''%s'' output a non-numeric value.\n', 'Reshape_To_TransposeLayer1055');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Reshape_To_TransposeLayer1055'));
            end
            x_model_22_dfl_Trans = dlarray(single(x_model_22_dfl_Trans), 'SSCB');
            x_model_22_Sigmoid_o = dlarray(single(x_model_22_Sigmoid_o), repmat('U', 1, max(2, x_model_22_Sigmoid_oNumDims)));
            x_model_22_Sigmoid_oNumDims = dlarray(ones(1,x_model_22_Sigmoid_oNumDims,'like',x_model_22_dfl_Trans), 'UU');
            if ~coder.target('MATLAB')
                x_model_22_dfl_Trans = extractdata(x_model_22_dfl_Trans);
                x_model_22_Sigmoid_o = extractdata(x_model_22_Sigmoid_o);
                x_model_22_Sigmoid_oNumDims = extractdata(x_model_22_Sigmoid_oNumDims);
            end
        end
    end
end

function [x_model_22_dfl_Trans, x_model_22_Sigmoid_o, x_model_22_dfl_TransNumDims, x_model_22_Sigmoid_oNumDims, state] = Reshape_To_TransposeFcn(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, x_model_22_Concat_ouNumDims, x_model_22_Concat_1_NumDims, x_model_22_Concat_2_NumDims, params, varargin)
%RESHAPE_TO_TRANSPOSEFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 14
%
% Variable names in this function are taken from the original ONNX file.
%
% [X_MODEL_22_DFL_TRANS, X_MODEL_22_SIGMOID_O] = Reshape_To_TransposeFcn(X_MODEL_22_CONCAT_OU, X_MODEL_22_CONCAT_1_, X_MODEL_22_CONCAT_2_, PARAMS)
%			- Evaluates the imported ONNX network RESHAPE_TO_TRANSPOSEFCN with input(s)
%			X_MODEL_22_CONCAT_OU, X_MODEL_22_CONCAT_1_, X_MODEL_22_CONCAT_2_ and the imported network parameters in PARAMS. Returns
%			network output(s) in X_MODEL_22_DFL_TRANS, X_MODEL_22_SIGMOID_O.
%
% [X_MODEL_22_DFL_TRANS, X_MODEL_22_SIGMOID_O, STATE] = Reshape_To_TransposeFcn(X_MODEL_22_CONCAT_OU, X_MODEL_22_CONCAT_1_, X_MODEL_22_CONCAT_2_, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Reshape_To_TransposeFcn(X_MODEL_22_CONCAT_OU, X_MODEL_22_CONCAT_1_, X_MODEL_22_CONCAT_2_, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
%			- Specifies additional name-value pairs described below:
%
% 'Training'
% 			Boolean indicating whether the network is being evaluated for
%			prediction or training. If TRAINING is true, state variables
%			will be updated.
%
% 'InputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			 between the dimensions of the input data and the dimensions of
%			the ONNX model input. For example, the permutation from HWCN
%			(MATLAB standard) to NCHW (ONNX standard) uses the vector
%			[4 3 1 2]. See the documentation for IMPORTONNXFUNCTION for
%			more information about automatic permutation.
%
%			'none' - Input(s) are passed in the ONNX model format. See 'Inputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between input data dimensions and the expected
%			ONNX input dimensions.%
%			cell array - If the network has multiple inputs, each cell
%			contains 'auto', 'none', or a numeric vector.
%
% 'OutputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			between the dimensions of the output and a conventional MATLAB
%			dimension ordering. For example, the permutation from NC (ONNX
%			standard) to CN (MATLAB standard) uses the vector [2 1]. See
%			the documentation for IMPORTONNXFUNCTION for more information
%			about automatic permutation.
%
%			'none' - Return output(s) as given by the ONNX model. See 'Outputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between the ONNX output dimensions and the
%			desired output dimensions.%
%			cell array - If the network has multiple outputs, each cell
%			contains 'auto', 'none' or a numeric vector.
%
% Inputs:
% -------
% X_MODEL_22_CONCAT_OU, X_MODEL_22_CONCAT_1_, X_MODEL_22_CONCAT_2_
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  X_MODEL_22_CONCAT_OU:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  X_MODEL_22_CONCAT_1_:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  X_MODEL_22_CONCAT_2_:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%			  By default, the function will try to permute the input(s)
%			  into this dimension ordering. If the default is incorrect,
%			  use the 'InputDataPermutation' argument to control the
%			  permutation.
%
%
% PARAMS	- Network parameters returned by 'importONNXFunction'.
%
%
% Outputs:
% --------
% X_MODEL_22_DFL_TRANS, X_MODEL_22_SIGMOID_O
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  X_MODEL_22_DFL_TRANS:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  X_MODEL_22_SIGMOID_O:		[1, 1]				Type: FLOAT
%			  By default, the function will try to permute the output(s)
%			  from this dimension ordering into a conventional MATLAB
%			  ordering. If the default is incorrect, use the
%			  'OutputDataPermutation' argument to control the permutation.
%
% STATE		- (Optional) State variables. When TRAINING is true, these will
% 			  have been updated from the original values in PARAMS.State.
%
%
%  See also importONNXFunction

% Preprocess the input data and arguments:
[x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'x_model_22_Concat_ou', 'x_model_22_Concat_1_', 'x_model_22_Concat_2_'}, {x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_}, [x_model_22_Concat_ouNumDims x_model_22_Concat_1_NumDims x_model_22_Concat_2_NumDims]);
% Call the top-level graph function:
[x_model_22_dfl_Trans, x_model_22_Sigmoid_o, x_model_22_dfl_TransNumDims, x_model_22_Sigmoid_oNumDims, state] = Reshape_To_TransposeGraph1048(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, NumDims.x_model_22_Concat_ou, NumDims.x_model_22_Concat_1_, NumDims.x_model_22_Concat_2_, Vars, NumDims, Training, params.State);
% Postprocess the output data
[x_model_22_dfl_Trans, x_model_22_Sigmoid_o] = postprocessOutput(x_model_22_dfl_Trans, x_model_22_Sigmoid_o, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [x_model_22_dfl_Trans, x_model_22_Sigmoid_o, x_model_22_dfl_TransNumDims1053, x_model_22_Sigmoid_oNumDims1054, state] = Reshape_To_TransposeGraph1048(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, x_model_22_Concat_ouNumDims1050, x_model_22_Concat_1_NumDims1051, x_model_22_Concat_2_NumDims1052, Vars, NumDims, Training, state)
% Function implementing the graph 'Reshape_To_TransposeGraph1048'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.x_model_22_Concat_ou = x_model_22_Concat_ou;
NumDims.x_model_22_Concat_ou = x_model_22_Concat_ouNumDims1050;
Vars.x_model_22_Concat_1_ = x_model_22_Concat_1_;
NumDims.x_model_22_Concat_1_ = x_model_22_Concat_1_NumDims1051;
Vars.x_model_22_Concat_2_ = x_model_22_Concat_2_;
NumDims.x_model_22_Concat_2_ = x_model_22_Concat_2_NumDims1052;

% Execute the operators:
% Reshape:
[shape, NumDims.x_model_22_Reshape_o] = prepareReshapeArgs(Vars.x_model_22_Concat_ou, Vars.x_model_22_Constan_6, NumDims.x_model_22_Concat_ou, 0);
Vars.x_model_22_Reshape_o = reshape(Vars.x_model_22_Concat_ou, shape{:});

% Reshape:
[shape, NumDims.x_model_22_Reshape_1] = prepareReshapeArgs(Vars.x_model_22_Concat_1_, Vars.x_model_22_Constan_6, NumDims.x_model_22_Concat_1_, 0);
Vars.x_model_22_Reshape_1 = reshape(Vars.x_model_22_Concat_1_, shape{:});

% Reshape:
[shape, NumDims.x_model_22_Reshape_2] = prepareReshapeArgs(Vars.x_model_22_Concat_2_, Vars.x_model_22_Constan_6, NumDims.x_model_22_Concat_2_, 0);
Vars.x_model_22_Reshape_2 = reshape(Vars.x_model_22_Concat_2_, shape{:});

% Concat:
[Vars.x_model_22_Concat_3_, NumDims.x_model_22_Concat_3_] = onnxConcat(2, {Vars.x_model_22_Reshape_o, Vars.x_model_22_Reshape_1, Vars.x_model_22_Reshape_2}, [NumDims.x_model_22_Reshape_o, NumDims.x_model_22_Reshape_1, NumDims.x_model_22_Reshape_2]);

% Split:
[Vars.x_model_22_Split_out, Vars.x_model_22_Split_o_1, NumDims.x_model_22_Split_out, NumDims.x_model_22_Split_o_1] = onnxSplit13(Vars.x_model_22_Concat_3_, 1, Vars.onnx__Split_418, 2, NumDims.x_model_22_Concat_3_);

% Reshape:
[shape, NumDims.x_model_22_dfl_Res_1] = prepareReshapeArgs(Vars.x_model_22_Split_out, Vars.x_model_22_dfl_Con_1, NumDims.x_model_22_Split_out, 0);
Vars.x_model_22_dfl_Res_1 = reshape(Vars.x_model_22_Split_out, shape{:});

% Sigmoid:
Vars.x_model_22_Sigmoid_o = sigmoid(Vars.x_model_22_Split_o_1);
NumDims.x_model_22_Sigmoid_o = NumDims.x_model_22_Split_o_1;

% Transpose:
[perm, NumDims.x_model_22_dfl_Trans] = prepareTransposeArgs(Vars.TransposePerm1049, NumDims.x_model_22_dfl_Res_1);
if ~isempty(perm)
    Vars.x_model_22_dfl_Trans = permute(Vars.x_model_22_dfl_Res_1, perm);
end

% Set graph output arguments from Vars and NumDims:
x_model_22_dfl_Trans = Vars.x_model_22_dfl_Trans;
x_model_22_dfl_TransNumDims1053 = NumDims.x_model_22_dfl_Trans;
x_model_22_Sigmoid_o = Vars.x_model_22_Sigmoid_o;
x_model_22_Sigmoid_oNumDims1054 = NumDims.x_model_22_Sigmoid_o;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, numDataOutputs, params, varargin)
% Function to validate inputs to Reshape_To_TransposeFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'x_model_22_Concat_ou', isValidArrayInput);
addRequired(p, 'x_model_22_Concat_1_', isValidArrayInput);
addRequired(p, 'x_model_22_Concat_2_', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, params, varargin{:});
inputDataPerms = p.Results.InputDataPermutation;
outputDataPerms = p.Results.OutputDataPermutation;
Training = p.Results.Training;
if isnumeric(inputDataPerms)
    inputDataPerms = {inputDataPerms};
end
if isstring(inputDataPerms) && isscalar(inputDataPerms) || ischar(inputDataPerms)
    inputDataPerms = repmat({inputDataPerms},1,3);
end
if isnumeric(outputDataPerms)
    outputDataPerms = {outputDataPerms};
end
if isstring(outputDataPerms) && isscalar(outputDataPerms) || ischar(outputDataPerms)
    outputDataPerms = repmat({outputDataPerms},1,numDataOutputs);
end
end

function [x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, 2, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_}));
% Make the input variables into unlabelled dlarrays:
x_model_22_Concat_ou = makeUnlabeledDlarray(x_model_22_Concat_ou);
x_model_22_Concat_1_ = makeUnlabeledDlarray(x_model_22_Concat_1_);
x_model_22_Concat_2_ = makeUnlabeledDlarray(x_model_22_Concat_2_);
% Permute inputs if requested:
x_model_22_Concat_ou = permuteInputVar(x_model_22_Concat_ou, inputDataPerms{1}, 4);
x_model_22_Concat_1_ = permuteInputVar(x_model_22_Concat_1_, inputDataPerms{2}, 4);
x_model_22_Concat_2_ = permuteInputVar(x_model_22_Concat_2_, inputDataPerms{3}, 4);
end

function [x_model_22_dfl_Trans, x_model_22_Sigmoid_o] = postprocessOutput(x_model_22_dfl_Trans, x_model_22_Sigmoid_o, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(x_model_22_dfl_Trans)
        x_model_22_dfl_Trans = extractdata(x_model_22_dfl_Trans);
    end
    if isdlarray(x_model_22_Sigmoid_o)
        x_model_22_Sigmoid_o = extractdata(x_model_22_Sigmoid_o);
    end
end
% Permute outputs if requested:
x_model_22_dfl_Trans = permuteOutputVar(x_model_22_dfl_Trans, outputDataPerms{1}, 4);
end


%% dlarray functions implementing ONNX operators:

function [Y, numDimsY] = onnxConcat(ONNXAxis, XCell, numDimsXArray)
% Concatentation that treats all empties the same. Necessary because
% dlarray.cat does not allow, for example, cat(1, 1x1, 1x0) because the
% second dimension sizes do not match.
numDimsY = numDimsXArray(1);
XCell(cellfun(@isempty, XCell)) = [];
if isempty(XCell)
    Y = dlarray([]);
else
    if ONNXAxis<0
        ONNXAxis = ONNXAxis + numDimsY;
    end
    DLTAxis = numDimsY - ONNXAxis;
    Y = cat(DLTAxis, XCell{:});
end
end

function varargout = onnxSplit13(X, ONNXaxis, splits, numSplits, numDimsX)
% Implements the ONNX Split operator

% ONNXaxis is origin 0. splits is a vector of the lengths of each segment.
% If splits is empty, instead split into segments of equal length.
if ONNXaxis<0
    ONNXaxis = ONNXaxis + numDimsX;
end
DLTAxis = numDimsX - ONNXaxis;
if isempty(splits)
    C       = size(X, DLTAxis);
    sz      = floor(C/numSplits);
    splits	= repmat(sz, 1, numSplits);
else
    splits = extractdata(splits);
end
S      = struct;
S.type = '()';
S.subs = repmat({':'}, 1, ndims(X));
splitIndices = [0 cumsum(splits(:)')];
numY = numel(splitIndices)-1;
for i = 1:numY
    from            = splitIndices(i) + 1;
    to              = splitIndices(i+1);
    S.subs{DLTAxis}	= from:to;
    % The first numY outputs are the Y's. The second numY outputs are their
    % numDims. We assume all the outputs of Split have the same numDims as
    % the input.
    varargout{i}        = subsref(X, S);
    varargout{i + numY} = numDimsX;
end
end

function [DLTShape, numDimsY] = prepareReshapeArgs(X, ONNXShape, numDimsX, allowzero)
% Prepares arguments for implementing the ONNX Reshape operator
ONNXShape = flip(extractdata(ONNXShape));            % First flip the shape to make it correspond to the dimensions of X.
% In ONNX, 0 means "unchanged" if allowzero is false, and -1 means "infer". In DLT, there is no
% "unchanged", and [] means "infer".
DLTShape = num2cell(ONNXShape);                      % Make a cell array so we can include [].
% Replace zeros with the actual size if allowzero is true
if any(ONNXShape==0) && allowzero==0
    i0 = find(ONNXShape==0);
    DLTShape(i0) = num2cell(size(X, numDimsX - numel(ONNXShape) + i0));  % right-align the shape vector and dims
end
if any(ONNXShape == -1)
    % Replace -1 with []
    i = ONNXShape == -1;
    DLTShape{i} = [];
end
if numel(DLTShape)==1
    DLTShape = [DLTShape 1];
end
numDimsY = numel(ONNXShape);
end

function [perm, numDimsA] = prepareTransposeArgs(ONNXPerm, numDimsA)
% Prepares arguments for implementing the ONNX Transpose operator
if numDimsA <= 1        % Tensors of numDims 0 or 1 are unchanged by ONNX Transpose.
    perm = [];
else
    if isempty(ONNXPerm)        % Empty ONNXPerm means reverse the dimensions.
        perm = numDimsA:-1:1;
    else
        perm = numDimsA-flip(ONNXPerm);
    end
end
end

%% Utility functions:

function s = appendStructs(varargin)
% s = appendStructs(s1, s2,...). Assign all fields in s1, s2,... into s.
if isempty(varargin)
    s = struct;
else
    s = varargin{1};
    for i = 2:numel(varargin)
        fromstr = varargin{i};
        fs = fieldnames(fromstr);
        for j = 1:numel(fs)
            s.(fs{j}) = fromstr.(fs{j});
        end
    end
end
end

function checkInputSize(inputShape, expectedShape, inputName)

if numel(expectedShape)==0
    % The input is a scalar
    if ~isequal(inputShape, [1 1])
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, "[1,1]", inputSizeStr));
    end
elseif numel(expectedShape)==1
    % The input is a vector
    if ~shapeIsColumnVector(inputShape) || ~iSizesMatch({inputShape(1)}, expectedShape)
        expectedShape{2} = 1;
        expectedSizeStr = makeSizeString(expectedShape);
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
else
    % The input has 2 dimensions or more
    
    % The input dimensions have been reversed; flip them back to compare to the
    % expected ONNX shape.
    inputShape = fliplr(inputShape);
    
    % If the expected shape has fewer dims than the input shape, error.
    if numel(expectedShape) < numel(inputShape)
        expectedSizeStr = strjoin(["[", strjoin(string(expectedShape), ","), "]"], "");
        error(message('nnet_cnn_onnx:onnx:InputHasGreaterNDims', inputName, expectedSizeStr));
    end
    
    % Prepad the input shape with trailing ones up to the number of elements in
    % expectedShape
    inputShape = num2cell([ones(1, numel(expectedShape) - length(inputShape)) inputShape]);
    
    % Find the number of variable size dimensions in the expected shape
    numVariableInputs = sum(cellfun(@(x) isa(x, 'char') || isa(x, 'string'), expectedShape));
    
    % Find the number of input dimensions that are not in the expected shape
    % and cannot be represented by a variable dimension
    nonMatchingInputDims = setdiff(string(inputShape), string(expectedShape));
    numNonMatchingInputDims  = numel(nonMatchingInputDims) - numVariableInputs;
    
    expectedSizeStr = makeSizeString(expectedShape);
    inputSizeStr = makeSizeString(inputShape);
    if numNonMatchingInputDims == 0 && ~iSizesMatch(inputShape, expectedShape)
        % The actual and expected input dimensions match, but in
        % a different order. The input needs to be permuted.
        error(message('nnet_cnn_onnx:onnx:InputNeedsPermute',inputName, expectedSizeStr, inputSizeStr));
    elseif numNonMatchingInputDims > 0
        % The actual and expected input sizes do not match.
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
end
end

function doesMatch = iSizesMatch(inputShape, expectedShape)
% Check whether the input and expected shapes match, in order.
% Size elements match if (1) the elements are equal, or (2) the expected
% size element is a variable (represented by a character vector or string)
doesMatch = true;
for i=1:numel(inputShape)
    if ~(isequal(inputShape{i},expectedShape{i}) || ischar(expectedShape{i}) || isstring(expectedShape{i}))
        doesMatch = false;
        return
    end
end
end

function sizeStr = makeSizeString(shape)
sizeStr = strjoin(["[", strjoin(string(shape), ","), "]"], "");
end

function isVec = shapeIsColumnVector(shape)
if numel(shape) == 2 && shape(2) == 1
    isVec = true;
else
    isVec = false;
end
end
function X = makeUnlabeledDlarray(X)
% Make numeric X into an unlabelled dlarray
if isa(X, 'dlarray')
    X = stripdims(X);
elseif isnumeric(X)
    if isinteger(X)
        % Make ints double so they can combine with anything without
        % reducing precision
        X = double(X);
    end
    X = dlarray(X);
end
end

function [Vars, NumDims] = packageVariables(params, inputNames, inputValues, inputNumDims)
% inputNames, inputValues are cell arrays. inputRanks is a numeric vector.
Vars = appendStructs(params.Learnables, params.Nonlearnables, params.State);
NumDims = params.NumDimensions;
% Add graph inputs
for i = 1:numel(inputNames)
    Vars.(inputNames{i}) = inputValues{i};
    NumDims.(inputNames{i}) = inputNumDims(i);
end
end

function X = permuteInputVar(X, userDataPerm, onnxNDims)
% Returns reverse-ONNX ordering
if onnxNDims == 0
    return;
elseif onnxNDims == 1 && isvector(X)
    X = X(:);
    return;
elseif isnumeric(userDataPerm)
    % Permute into reverse ONNX ordering
    if numel(userDataPerm) ~= onnxNDims
        error(message('nnet_cnn_onnx:onnx:InputPermutationSize', numel(userDataPerm), onnxNDims));
    end
    perm = fliplr(userDataPerm);
elseif isequal(userDataPerm, 'auto') && onnxNDims == 4
    % Permute MATLAB HWCN to reverse onnx (WHCN)
    perm = [2 1 3 4];
elseif isequal(userDataPerm, 'as-is')
    % Do not permute the input
    perm = 1:ndims(X);
else
    % userDataPerm is either 'none' or 'auto' with no default, which means
    % it's already in onnx ordering, so just make it reverse onnx
    perm = max(2,onnxNDims):-1:1;
end
X = permute(X, perm);
end

function Y = permuteOutputVar(Y, userDataPerm, onnxNDims)
switch onnxNDims
    case 0
        perm = [];
    case 1
        if isnumeric(userDataPerm)
            % Use the user's permutation because Y is a column vector which
            % already matches ONNX.
            perm = userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            % Treat the 1D onnx vector as a 2D column and transpose it
            perm = [2 1];
        else
            % userDataPerm is 'none'. Leave Y alone because it already
            % matches onnx.
            perm = [];
        end
    otherwise
        % ndims >= 2
        if isnumeric(userDataPerm)
            % Use the inverse of the user's permutation. This is not just the
            % flip of the permutation vector.
            perm = onnxNDims + 1 - userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            if onnxNDims == 2
                % Permute reverse ONNX CN to DLT CN (do nothing)
                perm = [];
            elseif onnxNDims == 4
                % Permute reverse onnx (WHCN) to MATLAB HWCN
                perm = [2 1 3 4];
            else
                % User wants the output in ONNX ordering, so just reverse it from
                % reverse onnx
                perm = onnxNDims:-1:1;
            end
        elseif isequal(userDataPerm, 'as-is')
            % Do not permute the input
            perm = 1:ndims(Y);
        else
            % userDataPerm is 'none', so just make it reverse onnx
            perm = onnxNDims:-1:1;
        end
end
if ~isempty(perm)
    Y = permute(Y, perm);
end
end

function s = updateStruct(s, t)
% Set all existing fields in s from fields in t, ignoring extra fields in t.
for name = transpose(fieldnames(s))
    s.(name{1}) = t.(name{1});
end
end
