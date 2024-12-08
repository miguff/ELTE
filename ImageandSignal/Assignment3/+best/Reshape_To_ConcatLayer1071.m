classdef Reshape_To_ConcatLayer1071 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    
    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    
    properties (Learnable)
    end
    
    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end
    
    methods
        function this = Reshape_To_ConcatLayer1071(name, onnxParams)
            this.Name = name;
            this.NumInputs = 9;
            this.OutputNames = {'output0'};
            this.ONNXParams = onnxParams;
        end
        
        function [output0] = predict(this, x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_ConcaNumDims, x_model_22_Unsque_2NumDims, x_model_22_TransposeNumDims, x_model_22_Sigmoid_oNumDims)
            if isdlarray(x_model_22_dfl_conv_)
                x_model_22_dfl_conv_ = stripdims(x_model_22_dfl_conv_);
            end
            if isdlarray(x_model_22_dfl_Conca)
                x_model_22_dfl_Conca = stripdims(x_model_22_dfl_Conca);
            end
            if isdlarray(x_model_22_Unsque_2)
                x_model_22_Unsque_2 = stripdims(x_model_22_Unsque_2);
            end
            if isdlarray(x_model_22_Transpose)
                x_model_22_Transpose = stripdims(x_model_22_Transpose);
            end
            if isdlarray(x_model_22_Sigmoid_o)
                x_model_22_Sigmoid_o = stripdims(x_model_22_Sigmoid_o);
            end
            x_model_22_dfl_conv_NumDims = 4;
            x_model_22_dfl_ConcaNumDims = numel(x_model_22_dfl_ConcaNumDims);
            x_model_22_Unsque_2NumDims = numel(x_model_22_Unsque_2NumDims);
            x_model_22_TransposeNumDims = numel(x_model_22_TransposeNumDims);
            x_model_22_Sigmoid_oNumDims = numel(x_model_22_Sigmoid_oNumDims);
            onnxParams = this.ONNXParams;
            [output0, output0NumDims] = Reshape_To_ConcatFcn(x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_conv_NumDims, x_model_22_dfl_ConcaNumDims, x_model_22_Unsque_2NumDims, x_model_22_TransposeNumDims, x_model_22_Sigmoid_oNumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[4 3 1 2], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A), {output0}))
                fprintf('Runtime error in network. The custom layer ''%s'' output a non-numeric value.\n', 'Reshape_To_ConcatLayer1071');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Reshape_To_ConcatLayer1071'));
            end
            output0 = dlarray(single(output0), repmat('U', 1, max(2, output0NumDims)));
            if ~coder.target('MATLAB')
                output0 = extractdata(output0);
            end
        end
        
        function [output0] = forward(this, x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_ConcaNumDims, x_model_22_Unsque_2NumDims, x_model_22_TransposeNumDims, x_model_22_Sigmoid_oNumDims)
            if isdlarray(x_model_22_dfl_conv_)
                x_model_22_dfl_conv_ = stripdims(x_model_22_dfl_conv_);
            end
            if isdlarray(x_model_22_dfl_Conca)
                x_model_22_dfl_Conca = stripdims(x_model_22_dfl_Conca);
            end
            if isdlarray(x_model_22_Unsque_2)
                x_model_22_Unsque_2 = stripdims(x_model_22_Unsque_2);
            end
            if isdlarray(x_model_22_Transpose)
                x_model_22_Transpose = stripdims(x_model_22_Transpose);
            end
            if isdlarray(x_model_22_Sigmoid_o)
                x_model_22_Sigmoid_o = stripdims(x_model_22_Sigmoid_o);
            end
            x_model_22_dfl_conv_NumDims = 4;
            x_model_22_dfl_ConcaNumDims = numel(x_model_22_dfl_ConcaNumDims);
            x_model_22_Unsque_2NumDims = numel(x_model_22_Unsque_2NumDims);
            x_model_22_TransposeNumDims = numel(x_model_22_TransposeNumDims);
            x_model_22_Sigmoid_oNumDims = numel(x_model_22_Sigmoid_oNumDims);
            onnxParams = this.ONNXParams;
            [output0, output0NumDims] = Reshape_To_ConcatFcn(x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_conv_NumDims, x_model_22_dfl_ConcaNumDims, x_model_22_Unsque_2NumDims, x_model_22_TransposeNumDims, x_model_22_Sigmoid_oNumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[4 3 1 2], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A), {output0}))
                fprintf('Runtime error in network. The custom layer ''%s'' output a non-numeric value.\n', 'Reshape_To_ConcatLayer1071');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Reshape_To_ConcatLayer1071'));
            end
            output0 = dlarray(single(output0), repmat('U', 1, max(2, output0NumDims)));
            if ~coder.target('MATLAB')
                output0 = extractdata(output0);
            end
        end
    end
end

function [output0, output0NumDims, state] = Reshape_To_ConcatFcn(x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_conv_NumDims, x_model_22_dfl_ConcaNumDims, x_model_22_Unsque_2NumDims, x_model_22_TransposeNumDims, x_model_22_Sigmoid_oNumDims, params, varargin)
%RESHAPE_TO_CONCATFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 14
%
% Variable names in this function are taken from the original ONNX file.
%
% [OUTPUT0] = Reshape_To_ConcatFcn(X_MODEL_22_DFL_CONV_, X_MODEL_22_DFL_CONCA, X_MODEL_22_UNSQUE_2, X_MODEL_22_TRANSPOSE, X_MODEL_22_SIGMOID_O, PARAMS)
%			- Evaluates the imported ONNX network RESHAPE_TO_CONCATFCN with input(s)
%			X_MODEL_22_DFL_CONV_, X_MODEL_22_DFL_CONCA, X_MODEL_22_UNSQUE_2, X_MODEL_22_TRANSPOSE, X_MODEL_22_SIGMOID_O and the imported network parameters in PARAMS. Returns
%			network output(s) in OUTPUT0.
%
% [OUTPUT0, STATE] = Reshape_To_ConcatFcn(X_MODEL_22_DFL_CONV_, X_MODEL_22_DFL_CONCA, X_MODEL_22_UNSQUE_2, X_MODEL_22_TRANSPOSE, X_MODEL_22_SIGMOID_O, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Reshape_To_ConcatFcn(X_MODEL_22_DFL_CONV_, X_MODEL_22_DFL_CONCA, X_MODEL_22_UNSQUE_2, X_MODEL_22_TRANSPOSE, X_MODEL_22_SIGMOID_O, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
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
% X_MODEL_22_DFL_CONV_, X_MODEL_22_DFL_CONCA, X_MODEL_22_UNSQUE_2, X_MODEL_22_TRANSPOSE, X_MODEL_22_SIGMOID_O
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  X_MODEL_22_DFL_CONV_:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  X_MODEL_22_DFL_CONCA:		[1, 1]				Type: FLOAT
%				  X_MODEL_22_UNSQUE_2:		[1, 1]				Type: FLOAT
%				  X_MODEL_22_TRANSPOSE:		[1, 1]				Type: FLOAT
%				  X_MODEL_22_SIGMOID_O:		[1, 1]				Type: FLOAT
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
% OUTPUT0
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  OUTPUT0:		[batch, 5, (floor(floor(floor(height/2 - 1/2)/2)/2) + 1)*(floor(floor(floor(width/2 - 1/2)/2)/2) + 1) + (floor(floor(floor(floor(height/2 - 1/2)/2)/2)/2) + 1)*(floor(floor(floor(floor(width/2 - 1/2)/2)/2)/2) + 1) + (floor(floor(floor(floor(floor(height/2 - 1/2)/2)/2)/2)/2) + 1)*(floor(floor(floor(floor(floor(width/2 - 1/2)/2)/2)/2)/2) + 1)]				Type: FLOAT
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
[x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'x_model_22_dfl_conv_', 'x_model_22_dfl_Conca', 'x_model_22_Unsque_2', 'x_model_22_Transpose', 'x_model_22_Sigmoid_o'}, {x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o}, [x_model_22_dfl_conv_NumDims x_model_22_dfl_ConcaNumDims x_model_22_Unsque_2NumDims x_model_22_TransposeNumDims x_model_22_Sigmoid_oNumDims]);
% Call the top-level graph function:
[output0, output0NumDims, state] = Reshape_To_ConcatGraph1064(x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, NumDims.x_model_22_dfl_conv_, NumDims.x_model_22_dfl_Conca, NumDims.x_model_22_Unsque_2, NumDims.x_model_22_Transpose, NumDims.x_model_22_Sigmoid_o, Vars, NumDims, Training, params.State);
% Postprocess the output data
[output0] = postprocessOutput(output0, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [output0, output0NumDims1070, state] = Reshape_To_ConcatGraph1064(x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_conv_NumDims1065, x_model_22_dfl_ConcaNumDims1066, x_model_22_Unsque_2NumDims1067, x_model_22_TransposeNumDims1068, x_model_22_Sigmoid_oNumDims1069, Vars, NumDims, Training, state)
% Function implementing the graph 'Reshape_To_ConcatGraph1064'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.x_model_22_dfl_conv_ = x_model_22_dfl_conv_;
NumDims.x_model_22_dfl_conv_ = x_model_22_dfl_conv_NumDims1065;
Vars.x_model_22_dfl_Conca = x_model_22_dfl_Conca;
NumDims.x_model_22_dfl_Conca = x_model_22_dfl_ConcaNumDims1066;
Vars.x_model_22_Unsque_2 = x_model_22_Unsque_2;
NumDims.x_model_22_Unsque_2 = x_model_22_Unsque_2NumDims1067;
Vars.x_model_22_Transpose = x_model_22_Transpose;
NumDims.x_model_22_Transpose = x_model_22_TransposeNumDims1068;
Vars.x_model_22_Sigmoid_o = x_model_22_Sigmoid_o;
NumDims.x_model_22_Sigmoid_o = x_model_22_Sigmoid_oNumDims1069;

% Execute the operators:
% Reshape:
[shape, NumDims.x_model_22_dfl_Resha] = prepareReshapeArgs(Vars.x_model_22_dfl_conv_, Vars.x_model_22_dfl_Conca, NumDims.x_model_22_dfl_conv_, 0);
Vars.x_model_22_dfl_Resha = reshape(Vars.x_model_22_dfl_conv_, shape{:});

% Slice:
[Indices, NumDims.x_model_22_Slice_out] = prepareSliceArgs(Vars.x_model_22_dfl_Resha, Vars.onnx__Unsqueeze_396, Vars.x_model_22_Mul_3_out, Vars.x_model_22_Consta_10, '', NumDims.x_model_22_dfl_Resha);
Vars.x_model_22_Slice_out = subsref(Vars.x_model_22_dfl_Resha, Indices);

% Slice:
[Indices, NumDims.x_model_22_Slice_1_o] = prepareSliceArgs(Vars.x_model_22_dfl_Resha, Vars.x_model_22_Mul_3_out, Vars.x_model_22_dfl_Const, Vars.x_model_22_Consta_10, '', NumDims.x_model_22_dfl_Resha);
Vars.x_model_22_Slice_1_o = subsref(Vars.x_model_22_dfl_Resha, Indices);

% Sub:
Vars.x_model_22_Sub_outpu = Vars.x_model_22_Unsque_2 - Vars.x_model_22_Slice_out;
NumDims.x_model_22_Sub_outpu = max(NumDims.x_model_22_Unsque_2, NumDims.x_model_22_Slice_out);

% Add:
Vars.x_model_22_Add_10_ou = Vars.x_model_22_Unsque_2 + Vars.x_model_22_Slice_1_o;
NumDims.x_model_22_Add_10_ou = max(NumDims.x_model_22_Unsque_2, NumDims.x_model_22_Slice_1_o);

% Add:
Vars.x_model_22_Add_11_ou = Vars.x_model_22_Sub_outpu + Vars.x_model_22_Add_10_ou;
NumDims.x_model_22_Add_11_ou = max(NumDims.x_model_22_Sub_outpu, NumDims.x_model_22_Add_10_ou);

% Sub:
Vars.x_model_22_Sub_1_out = Vars.x_model_22_Add_10_ou - Vars.x_model_22_Sub_outpu;
NumDims.x_model_22_Sub_1_out = max(NumDims.x_model_22_Add_10_ou, NumDims.x_model_22_Sub_outpu);

% Div:
Vars.x_model_22_Div_1_out = Vars.x_model_22_Add_11_ou ./ Vars.x_model_22_Consta_11;
NumDims.x_model_22_Div_1_out = max(NumDims.x_model_22_Add_11_ou, NumDims.x_model_22_Consta_11);

% Concat:
[Vars.x_model_22_Concat_24, NumDims.x_model_22_Concat_24] = onnxConcat(1, {Vars.x_model_22_Div_1_out, Vars.x_model_22_Sub_1_out}, [NumDims.x_model_22_Div_1_out, NumDims.x_model_22_Sub_1_out]);

% Mul:
Vars.x_model_22_Mul_5_out = Vars.x_model_22_Concat_24 .* Vars.x_model_22_Transpose;
NumDims.x_model_22_Mul_5_out = max(NumDims.x_model_22_Concat_24, NumDims.x_model_22_Transpose);

% Concat:
[Vars.output0, NumDims.output0] = onnxConcat(1, {Vars.x_model_22_Mul_5_out, Vars.x_model_22_Sigmoid_o}, [NumDims.x_model_22_Mul_5_out, NumDims.x_model_22_Sigmoid_o]);

% Set graph output arguments from Vars and NumDims:
output0 = Vars.output0;
output0NumDims1070 = NumDims.output0;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, numDataOutputs, params, varargin)
% Function to validate inputs to Reshape_To_ConcatFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'x_model_22_dfl_conv_', isValidArrayInput);
addRequired(p, 'x_model_22_dfl_Conca', isValidArrayInput);
addRequired(p, 'x_model_22_Unsque_2', isValidArrayInput);
addRequired(p, 'x_model_22_Transpose', isValidArrayInput);
addRequired(p, 'x_model_22_Sigmoid_o', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, params, varargin{:});
inputDataPerms = p.Results.InputDataPermutation;
outputDataPerms = p.Results.OutputDataPermutation;
Training = p.Results.Training;
if isnumeric(inputDataPerms)
    inputDataPerms = {inputDataPerms};
end
if isstring(inputDataPerms) && isscalar(inputDataPerms) || ischar(inputDataPerms)
    inputDataPerms = repmat({inputDataPerms},1,5);
end
if isnumeric(outputDataPerms)
    outputDataPerms = {outputDataPerms};
end
if isstring(outputDataPerms) && isscalar(outputDataPerms) || ischar(outputDataPerms)
    outputDataPerms = repmat({outputDataPerms},1,numDataOutputs);
end
end

function [x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, 1, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {x_model_22_dfl_conv_, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o}));
% Make the input variables into unlabelled dlarrays:
x_model_22_dfl_conv_ = makeUnlabeledDlarray(x_model_22_dfl_conv_);
x_model_22_dfl_Conca = makeUnlabeledDlarray(x_model_22_dfl_Conca);
x_model_22_Unsque_2 = makeUnlabeledDlarray(x_model_22_Unsque_2);
x_model_22_Transpose = makeUnlabeledDlarray(x_model_22_Transpose);
x_model_22_Sigmoid_o = makeUnlabeledDlarray(x_model_22_Sigmoid_o);
% Permute inputs if requested:
x_model_22_dfl_conv_ = permuteInputVar(x_model_22_dfl_conv_, inputDataPerms{1}, 4);
x_model_22_dfl_Conca = permuteInputVar(x_model_22_dfl_Conca, inputDataPerms{2}, 0);
x_model_22_Unsque_2 = permuteInputVar(x_model_22_Unsque_2, inputDataPerms{3}, 0);
x_model_22_Transpose = permuteInputVar(x_model_22_Transpose, inputDataPerms{4}, 0);
x_model_22_Sigmoid_o = permuteInputVar(x_model_22_Sigmoid_o, inputDataPerms{5}, 0);
end

function [output0] = postprocessOutput(output0, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(output0)
        output0 = extractdata(output0);
    end
end
% Permute outputs if requested:
output0 = permuteOutputVar(output0, outputDataPerms{1}, 3);
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

function [S, numDimsY] = prepareSliceArgs(X, Starts, Ends, Axes, Steps, numDimsX)
% Prepares arguments for implementing the ONNX Slice operator

% Starts, Ends and Axes are all origin 0. Axes refer to the ONNX dimension
% ordering, but X uses the reverse, DLT ordering. Starts, Ends, Axes, and
% Steps correspond positionally. Axes and Steps may be omitted, with
% defaults described in the ONNX spec.

% Set default Axes and Steps if not supplied
if isempty(Axes)
    Axes = 0:numDimsX-1;   % All axes
end
Axes(Axes<0) = Axes(Axes<0) + numDimsX; % Handle negative Axes.
if isempty(Steps)
    Steps = ones(1, numel(Starts));
end
% Init all dims to :
S.subs = repmat({':'}, 1, numDimsX);
S.type = '()';
% Set Starts and Ends for each axis
for i = 1:numel(Axes)
    DLTDim = numDimsX - Axes(i);                                               % The DLT dim is the reverse of the ONNX dim.
    % "If a negative value is passed for any of the start or end indices,
    % it represents number of elements before the end of that dimension."
    if Starts(i) < 0
        Starts(i) = size(X,DLTDim) + Starts(i);
    end
    if Ends(i) < 0
        Ends(i) = max(-1, size(X,DLTDim) + Ends(i));                        % The -1 case is when we're slicing backward and want to include 0.
    end
    % "If the value passed to start or end is larger than the n (the number
    % of elements in this dimension), it represents n."
    if Starts(i) > size(X,DLTDim)
        Starts(i) = size(X,DLTDim);
    end
    if Ends(i) > size(X,DLTDim)
        Ends(i) = size(X,DLTDim);
    end
    if Steps(i) > 0
        S.subs{DLTDim} = 1 + (Starts(i) : Steps(i) : Ends(i)-1);            % 1 + (Origin 0 indexing with end index excluded)
    else
        S.subs{DLTDim} = 1 + (Starts(i) : Steps(i) : Ends(i)+1);            % 1 + (Origin 0 indexing with end index excluded)
    end
end
numDimsY = numDimsX;
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
