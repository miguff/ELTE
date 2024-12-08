classdef Shape_To_TransposeLayer1063 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    
    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    
    properties (Learnable)
        onnx__Split_580
        onnx__Unsqueeze_396
        x_model_22_Consta_2
        x_model_22_Consta_3
        x_model_22_Consta_6
    end
    
    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end
    
    methods
        function this = Shape_To_TransposeLayer1063(name, onnxParams)
            this.Name = name;
            this.NumInputs = 3;
            this.NumOutputs = 9;
            this.OutputNames = {'x_model_22_dfl_Trans', 'x_model_22_dfl_Conca', 'x_model_22_Unsque_2', 'x_model_22_Transpose', 'x_model_22_Sigmoid_o', 'x_model_22_dfl_ConcaNumDims', 'x_model_22_Unsque_2NumDims', 'x_model_22_TransposeNumDims', 'x_model_22_Sigmoid_oNumDims'};
            this.ONNXParams = onnxParams;
            this.onnx__Split_580 = onnxParams.Learnables.onnx__Split_580;
            this.onnx__Unsqueeze_396 = onnxParams.Learnables.onnx__Unsqueeze_396;
            this.x_model_22_Consta_2 = onnxParams.Learnables.x_model_22_Consta_2;
            this.x_model_22_Consta_3 = onnxParams.Learnables.x_model_22_Consta_3;
            this.x_model_22_Consta_6 = onnxParams.Learnables.x_model_22_Consta_6;
        end
        
        function [x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_ConcaNumDims, x_model_22_Unsque_2NumDims, x_model_22_TransposeNumDims, x_model_22_Sigmoid_oNumDims] = predict(this, x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_)
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
            onnxParams.Learnables.onnx__Split_580 = this.onnx__Split_580;
            onnxParams.Learnables.onnx__Unsqueeze_396 = this.onnx__Unsqueeze_396;
            onnxParams.Learnables.x_model_22_Consta_2 = this.x_model_22_Consta_2;
            onnxParams.Learnables.x_model_22_Consta_3 = this.x_model_22_Consta_3;
            onnxParams.Learnables.x_model_22_Consta_6 = this.x_model_22_Consta_6;
            [x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_TransNumDims, x_model_22_dfl_ConcaNumDims, x_model_22_Unsque_2NumDims, x_model_22_TransposeNumDims, x_model_22_Sigmoid_oNumDims] = Shape_To_TransposeFcn(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, x_model_22_Concat_ouNumDims, x_model_22_Concat_1_NumDims, x_model_22_Concat_2_NumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[4 3 1 2], [4 3 1 2], [4 3 1 2], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[3 4 2 1], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A), {x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o}))
                fprintf('Runtime error in network. The custom layer ''%s'' output a non-numeric value.\n', 'Shape_To_TransposeLayer1063');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Shape_To_TransposeLayer1063'));
            end
            x_model_22_dfl_Trans = dlarray(single(x_model_22_dfl_Trans), 'SSCB');
            x_model_22_dfl_Conca = dlarray(single(x_model_22_dfl_Conca), repmat('U', 1, max(2, x_model_22_dfl_ConcaNumDims)));
            x_model_22_Unsque_2 = dlarray(single(x_model_22_Unsque_2), repmat('U', 1, max(2, x_model_22_Unsque_2NumDims)));
            x_model_22_Transpose = dlarray(single(x_model_22_Transpose), repmat('U', 1, max(2, x_model_22_TransposeNumDims)));
            x_model_22_Sigmoid_o = dlarray(single(x_model_22_Sigmoid_o), repmat('U', 1, max(2, x_model_22_Sigmoid_oNumDims)));
            x_model_22_dfl_ConcaNumDims = dlarray(ones(1,x_model_22_dfl_ConcaNumDims,'like',x_model_22_dfl_Trans), 'UU');
            x_model_22_Unsque_2NumDims = dlarray(ones(1,x_model_22_Unsque_2NumDims,'like',x_model_22_dfl_Trans), 'UU');
            x_model_22_TransposeNumDims = dlarray(ones(1,x_model_22_TransposeNumDims,'like',x_model_22_dfl_Trans), 'UU');
            x_model_22_Sigmoid_oNumDims = dlarray(ones(1,x_model_22_Sigmoid_oNumDims,'like',x_model_22_dfl_Trans), 'UU');
            if ~coder.target('MATLAB')
                x_model_22_dfl_Trans = extractdata(x_model_22_dfl_Trans);
                x_model_22_dfl_Conca = extractdata(x_model_22_dfl_Conca);
                x_model_22_Unsque_2 = extractdata(x_model_22_Unsque_2);
                x_model_22_Transpose = extractdata(x_model_22_Transpose);
                x_model_22_Sigmoid_o = extractdata(x_model_22_Sigmoid_o);
                x_model_22_dfl_ConcaNumDims = extractdata(x_model_22_dfl_ConcaNumDims);
                x_model_22_Unsque_2NumDims = extractdata(x_model_22_Unsque_2NumDims);
                x_model_22_TransposeNumDims = extractdata(x_model_22_TransposeNumDims);
                x_model_22_Sigmoid_oNumDims = extractdata(x_model_22_Sigmoid_oNumDims);
            end
        end
        
        function [x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_ConcaNumDims, x_model_22_Unsque_2NumDims, x_model_22_TransposeNumDims, x_model_22_Sigmoid_oNumDims] = forward(this, x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_)
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
            onnxParams.Learnables.onnx__Split_580 = this.onnx__Split_580;
            onnxParams.Learnables.onnx__Unsqueeze_396 = this.onnx__Unsqueeze_396;
            onnxParams.Learnables.x_model_22_Consta_2 = this.x_model_22_Consta_2;
            onnxParams.Learnables.x_model_22_Consta_3 = this.x_model_22_Consta_3;
            onnxParams.Learnables.x_model_22_Consta_6 = this.x_model_22_Consta_6;
            [x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_TransNumDims, x_model_22_dfl_ConcaNumDims, x_model_22_Unsque_2NumDims, x_model_22_TransposeNumDims, x_model_22_Sigmoid_oNumDims] = Shape_To_TransposeFcn(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, x_model_22_Concat_ouNumDims, x_model_22_Concat_1_NumDims, x_model_22_Concat_2_NumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[4 3 1 2], [4 3 1 2], [4 3 1 2], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[3 4 2 1], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is'], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A), {x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o}))
                fprintf('Runtime error in network. The custom layer ''%s'' output a non-numeric value.\n', 'Shape_To_TransposeLayer1063');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Shape_To_TransposeLayer1063'));
            end
            x_model_22_dfl_Trans = dlarray(single(x_model_22_dfl_Trans), 'SSCB');
            x_model_22_dfl_Conca = dlarray(single(x_model_22_dfl_Conca), repmat('U', 1, max(2, x_model_22_dfl_ConcaNumDims)));
            x_model_22_Unsque_2 = dlarray(single(x_model_22_Unsque_2), repmat('U', 1, max(2, x_model_22_Unsque_2NumDims)));
            x_model_22_Transpose = dlarray(single(x_model_22_Transpose), repmat('U', 1, max(2, x_model_22_TransposeNumDims)));
            x_model_22_Sigmoid_o = dlarray(single(x_model_22_Sigmoid_o), repmat('U', 1, max(2, x_model_22_Sigmoid_oNumDims)));
            x_model_22_dfl_ConcaNumDims = dlarray(ones(1,x_model_22_dfl_ConcaNumDims,'like',x_model_22_dfl_Trans), 'UU');
            x_model_22_Unsque_2NumDims = dlarray(ones(1,x_model_22_Unsque_2NumDims,'like',x_model_22_dfl_Trans), 'UU');
            x_model_22_TransposeNumDims = dlarray(ones(1,x_model_22_TransposeNumDims,'like',x_model_22_dfl_Trans), 'UU');
            x_model_22_Sigmoid_oNumDims = dlarray(ones(1,x_model_22_Sigmoid_oNumDims,'like',x_model_22_dfl_Trans), 'UU');
            if ~coder.target('MATLAB')
                x_model_22_dfl_Trans = extractdata(x_model_22_dfl_Trans);
                x_model_22_dfl_Conca = extractdata(x_model_22_dfl_Conca);
                x_model_22_Unsque_2 = extractdata(x_model_22_Unsque_2);
                x_model_22_Transpose = extractdata(x_model_22_Transpose);
                x_model_22_Sigmoid_o = extractdata(x_model_22_Sigmoid_o);
                x_model_22_dfl_ConcaNumDims = extractdata(x_model_22_dfl_ConcaNumDims);
                x_model_22_Unsque_2NumDims = extractdata(x_model_22_Unsque_2NumDims);
                x_model_22_TransposeNumDims = extractdata(x_model_22_TransposeNumDims);
                x_model_22_Sigmoid_oNumDims = extractdata(x_model_22_Sigmoid_oNumDims);
            end
        end
    end
end

function [x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_TransNumDims, x_model_22_dfl_ConcaNumDims, x_model_22_Unsque_2NumDims, x_model_22_TransposeNumDims, x_model_22_Sigmoid_oNumDims, state] = Shape_To_TransposeFcn(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, x_model_22_Concat_ouNumDims, x_model_22_Concat_1_NumDims, x_model_22_Concat_2_NumDims, params, varargin)
%SHAPE_TO_TRANSPOSEFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 14
%
% Variable names in this function are taken from the original ONNX file.
%
% [X_MODEL_22_DFL_TRANS, X_MODEL_22_DFL_CONCA, X_MODEL_22_UNSQUE_2, X_MODEL_22_TRANSPOSE, X_MODEL_22_SIGMOID_O] = Shape_To_TransposeFcn(X_MODEL_22_CONCAT_OU, X_MODEL_22_CONCAT_1_, X_MODEL_22_CONCAT_2_, PARAMS)
%			- Evaluates the imported ONNX network SHAPE_TO_TRANSPOSEFCN with input(s)
%			X_MODEL_22_CONCAT_OU, X_MODEL_22_CONCAT_1_, X_MODEL_22_CONCAT_2_ and the imported network parameters in PARAMS. Returns
%			network output(s) in X_MODEL_22_DFL_TRANS, X_MODEL_22_DFL_CONCA, X_MODEL_22_UNSQUE_2, X_MODEL_22_TRANSPOSE, X_MODEL_22_SIGMOID_O.
%
% [X_MODEL_22_DFL_TRANS, X_MODEL_22_DFL_CONCA, X_MODEL_22_UNSQUE_2, X_MODEL_22_TRANSPOSE, X_MODEL_22_SIGMOID_O, STATE] = Shape_To_TransposeFcn(X_MODEL_22_CONCAT_OU, X_MODEL_22_CONCAT_1_, X_MODEL_22_CONCAT_2_, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Shape_To_TransposeFcn(X_MODEL_22_CONCAT_OU, X_MODEL_22_CONCAT_1_, X_MODEL_22_CONCAT_2_, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
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
% X_MODEL_22_DFL_TRANS, X_MODEL_22_DFL_CONCA, X_MODEL_22_UNSQUE_2, X_MODEL_22_TRANSPOSE, X_MODEL_22_SIGMOID_O
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  X_MODEL_22_DFL_TRANS:		[Unknown, Unknown, Unknown, Unknown]				Type: FLOAT
%				  X_MODEL_22_DFL_CONCA:		[1, 1]				Type: FLOAT
%				  X_MODEL_22_UNSQUE_2:		[1, 1]				Type: FLOAT
%				  X_MODEL_22_TRANSPOSE:		[1, 1]				Type: FLOAT
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
[x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_TransNumDims, x_model_22_dfl_ConcaNumDims, x_model_22_Unsque_2NumDims, x_model_22_TransposeNumDims, x_model_22_Sigmoid_oNumDims, state] = Shape_To_TransposeGraph1048(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, NumDims.x_model_22_Concat_ou, NumDims.x_model_22_Concat_1_, NumDims.x_model_22_Concat_2_, Vars, NumDims, Training, params.State);
% Postprocess the output data
[x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o] = postprocessOutput(x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, x_model_22_dfl_TransNumDims1058, x_model_22_dfl_ConcaNumDims1059, x_model_22_Unsque_2NumDims1060, x_model_22_TransposeNumDims1061, x_model_22_Sigmoid_oNumDims1062, state] = Shape_To_TransposeGraph1048(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, x_model_22_Concat_ouNumDims1055, x_model_22_Concat_1_NumDims1056, x_model_22_Concat_2_NumDims1057, Vars, NumDims, Training, state)
% Function implementing the graph 'Shape_To_TransposeGraph1048'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.x_model_22_Concat_ou = x_model_22_Concat_ou;
NumDims.x_model_22_Concat_ou = x_model_22_Concat_ouNumDims1055;
Vars.x_model_22_Concat_1_ = x_model_22_Concat_1_;
NumDims.x_model_22_Concat_1_ = x_model_22_Concat_1_NumDims1056;
Vars.x_model_22_Concat_2_ = x_model_22_Concat_2_;
NumDims.x_model_22_Concat_2_ = x_model_22_Concat_2_NumDims1057;

% Execute the operators:
% Shape:
[Vars.x_model_22_Shape_out, NumDims.x_model_22_Shape_out] = onnxShape(Vars.x_model_22_Concat_ou, NumDims.x_model_22_Concat_ou);

% Gather:
[Vars.x_model_22_Gather_ou, NumDims.x_model_22_Gather_ou] = onnxGather(Vars.x_model_22_Shape_out, Vars.x_model_22_Consta_12, 0, NumDims.x_model_22_Shape_out, NumDims.x_model_22_Consta_12);

% Gather:
[Vars.x_model_22_Gather_1_, NumDims.x_model_22_Gather_1_] = onnxGather(Vars.x_model_22_Shape_out, Vars.x_model_22_Constant_, 0, NumDims.x_model_22_Shape_out, NumDims.x_model_22_Constant_);

% Gather:
[Vars.x_model_22_Gather_2_, NumDims.x_model_22_Gather_2_] = onnxGather(Vars.x_model_22_Shape_out, Vars.x_model_22_Consta_1, 0, NumDims.x_model_22_Shape_out, NumDims.x_model_22_Consta_1);

% Unsqueeze:
[shape, NumDims.x_model_22_Unsque_10] = prepareUnsqueezeArgs(Vars.x_model_22_Gather_ou, Vars.onnx__Unsqueeze_396, NumDims.x_model_22_Gather_ou);
Vars.x_model_22_Unsque_10 = reshape(Vars.x_model_22_Gather_ou, shape);

% Cast:
Vars.x_model_22_Cast_outp = single(Vars.x_model_22_Gather_2_);
NumDims.x_model_22_Cast_outp = NumDims.x_model_22_Gather_2_;

% Cast:
Vars.x_model_22_Cast_1_ou = single(Vars.x_model_22_Gather_1_);
NumDims.x_model_22_Cast_1_ou = NumDims.x_model_22_Gather_1_;

% Mul:
Vars.x_model_22_Mul_outpu = Vars.x_model_22_Gather_1_ .* Vars.x_model_22_Gather_2_;
NumDims.x_model_22_Mul_outpu = max(NumDims.x_model_22_Gather_1_, NumDims.x_model_22_Gather_2_);

% Concat:
[Vars.x_model_22_Concat_3_, NumDims.x_model_22_Concat_3_] = onnxConcat(0, {Vars.x_model_22_Unsque_10, Vars.x_model_22_Consta_5, Vars.x_model_22_Consta_9}, [NumDims.x_model_22_Unsque_10, NumDims.x_model_22_Consta_5, NumDims.x_model_22_Consta_9]);

% Range:
Vars.x_model_22_Range_out = dlarray(Vars.x_model_22_Consta_2:Vars.x_model_22_Consta_3:Vars.x_model_22_Cast_outp-sign(Vars.x_model_22_Consta_3))';
NumDims.x_model_22_Range_out = 1;

% Range:
Vars.x_model_22_Range_1_o = dlarray(Vars.x_model_22_Consta_2:Vars.x_model_22_Consta_3:Vars.x_model_22_Cast_1_ou-sign(Vars.x_model_22_Consta_3))';
NumDims.x_model_22_Range_1_o = 1;

% Unsqueeze:
[shape, NumDims.x_model_22_Unsque_5] = prepareUnsqueezeArgs(Vars.x_model_22_Mul_outpu, Vars.onnx__Unsqueeze_396, NumDims.x_model_22_Mul_outpu);
Vars.x_model_22_Unsque_5 = reshape(Vars.x_model_22_Mul_outpu, shape);

% Reshape:
[shape, NumDims.x_model_22_Reshape_o] = prepareReshapeArgs(Vars.x_model_22_Concat_ou, Vars.x_model_22_Concat_3_, NumDims.x_model_22_Concat_ou, 0);
Vars.x_model_22_Reshape_o = reshape(Vars.x_model_22_Concat_ou, shape{:});

% Add:
Vars.x_model_22_Add_outpu = Vars.x_model_22_Range_out + Vars.x_model_22_Consta_4;
NumDims.x_model_22_Add_outpu = max(NumDims.x_model_22_Range_out, NumDims.x_model_22_Consta_4);

% Add:
Vars.x_model_22_Add_1_out = Vars.x_model_22_Range_1_o + Vars.x_model_22_Consta_4;
NumDims.x_model_22_Add_1_out = max(NumDims.x_model_22_Range_1_o, NumDims.x_model_22_Consta_4);

% Concat:
[Vars.x_model_22_Concat_11, NumDims.x_model_22_Concat_11] = onnxConcat(0, {Vars.x_model_22_Unsque_5, Vars.x_model_22_Consta_8}, [NumDims.x_model_22_Unsque_5, NumDims.x_model_22_Consta_8]);

% ConstantOfShape:
[Vars.x_model_22_Constan_2, NumDims.x_model_22_Constan_2] = onnxConstantOfShape(Vars.ConstantOfShapeValue1049, Vars.x_model_22_Concat_11);

% Shape:
[Vars.x_model_22_Shape_3_o, NumDims.x_model_22_Shape_3_o] = onnxShape(Vars.x_model_22_Add_1_out, NumDims.x_model_22_Add_1_out);

% Shape:
[Vars.x_model_22_Shape_4_o, NumDims.x_model_22_Shape_4_o] = onnxShape(Vars.x_model_22_Add_outpu, NumDims.x_model_22_Add_outpu);

% Reshape:
[shape, NumDims.x_model_22_Reshape_5] = prepareReshapeArgs(Vars.x_model_22_Add_1_out, Vars.x_model_22_Concat_8_, NumDims.x_model_22_Add_1_out, 0);
Vars.x_model_22_Reshape_5 = reshape(Vars.x_model_22_Add_1_out, shape{:});

% Reshape:
[shape, NumDims.x_model_22_Reshape_6] = prepareReshapeArgs(Vars.x_model_22_Add_outpu, Vars.x_model_22_Concat_9_, NumDims.x_model_22_Add_outpu, 0);
Vars.x_model_22_Reshape_6 = reshape(Vars.x_model_22_Add_outpu, shape{:});

% Add:
Vars.x_model_22_Add_2_out = Vars.x_model_22_Constan_2 + Vars.x_model_22_Squeeze_o;
NumDims.x_model_22_Add_2_out = max(NumDims.x_model_22_Constan_2, NumDims.x_model_22_Squeeze_o);

% Concat:
[Vars.x_model_22_Concat_7_, NumDims.x_model_22_Concat_7_] = onnxConcat(0, {Vars.x_model_22_Shape_3_o, Vars.x_model_22_Shape_4_o}, [NumDims.x_model_22_Shape_3_o, NumDims.x_model_22_Shape_4_o]);

% Expand:
[shape, NumDims.x_model_22_Expand_ou] = prepareExpandArgs(Vars.x_model_22_Concat_7_);
Vars.x_model_22_Expand_ou = Vars.x_model_22_Reshape_5 + zeros(shape);

% Expand:
[shape, NumDims.x_model_22_Expand_1_] = prepareExpandArgs(Vars.x_model_22_Concat_7_);
Vars.x_model_22_Expand_1_ = Vars.x_model_22_Reshape_6 + zeros(shape);

% Unsqueeze:
[shape, NumDims.x_model_22_Unsque_3] = prepareUnsqueezeArgs(Vars.x_model_22_Expand_1_, Vars.x_model_22_Consta_6, NumDims.x_model_22_Expand_1_);
Vars.x_model_22_Unsque_3 = reshape(Vars.x_model_22_Expand_1_, shape);

% Unsqueeze:
[shape, NumDims.x_model_22_Unsque_4] = prepareUnsqueezeArgs(Vars.x_model_22_Expand_ou, Vars.x_model_22_Consta_6, NumDims.x_model_22_Expand_ou);
Vars.x_model_22_Unsque_4 = reshape(Vars.x_model_22_Expand_ou, shape);

% Concat:
[Vars.x_model_22_Concat_10, NumDims.x_model_22_Concat_10] = onnxConcat(-1, {Vars.x_model_22_Unsque_3, Vars.x_model_22_Unsque_4}, [NumDims.x_model_22_Unsque_3, NumDims.x_model_22_Unsque_4]);

% Reshape:
[shape, NumDims.x_model_22_Reshape_7] = prepareReshapeArgs(Vars.x_model_22_Concat_10, Vars.x_model_22_Consta_7, NumDims.x_model_22_Concat_10, 0);
Vars.x_model_22_Reshape_7 = reshape(Vars.x_model_22_Concat_10, shape{:});

% Reshape:
[shape, NumDims.x_model_22_Reshap_6] = prepareReshapeArgs(Vars.x_model_22_Concat_1_, Vars.x_model_22_Concat_3_, NumDims.x_model_22_Concat_1_, 0);
Vars.x_model_22_Reshap_6 = reshape(Vars.x_model_22_Concat_1_, shape{:});

% Shape:
[Vars.x_model_22_Shape_5_o, NumDims.x_model_22_Shape_5_o] = onnxShape(Vars.x_model_22_Concat_1_, NumDims.x_model_22_Concat_1_);

% Gather:
[Vars.x_model_22_Gather_3_, NumDims.x_model_22_Gather_3_] = onnxGather(Vars.x_model_22_Shape_5_o, Vars.x_model_22_Constant_, 0, NumDims.x_model_22_Shape_5_o, NumDims.x_model_22_Constant_);

% Gather:
[Vars.x_model_22_Gather_4_, NumDims.x_model_22_Gather_4_] = onnxGather(Vars.x_model_22_Shape_5_o, Vars.x_model_22_Consta_1, 0, NumDims.x_model_22_Shape_5_o, NumDims.x_model_22_Consta_1);

% Cast:
Vars.x_model_22_Cast_2_ou = single(Vars.x_model_22_Gather_4_);
NumDims.x_model_22_Cast_2_ou = NumDims.x_model_22_Gather_4_;

% Cast:
Vars.x_model_22_Cast_3_ou = single(Vars.x_model_22_Gather_3_);
NumDims.x_model_22_Cast_3_ou = NumDims.x_model_22_Gather_3_;

% Mul:
Vars.x_model_22_Mul_1_out = Vars.x_model_22_Gather_3_ .* Vars.x_model_22_Gather_4_;
NumDims.x_model_22_Mul_1_out = max(NumDims.x_model_22_Gather_3_, NumDims.x_model_22_Gather_4_);

% Range:
Vars.x_model_22_Range_2_o = dlarray(Vars.x_model_22_Consta_2:Vars.x_model_22_Consta_3:Vars.x_model_22_Cast_2_ou-sign(Vars.x_model_22_Consta_3))';
NumDims.x_model_22_Range_2_o = 1;

% Range:
Vars.x_model_22_Range_3_o = dlarray(Vars.x_model_22_Consta_2:Vars.x_model_22_Consta_3:Vars.x_model_22_Cast_3_ou-sign(Vars.x_model_22_Consta_3))';
NumDims.x_model_22_Range_3_o = 1;

% Unsqueeze:
[shape, NumDims.x_model_22_Unsque_8] = prepareUnsqueezeArgs(Vars.x_model_22_Mul_1_out, Vars.onnx__Unsqueeze_396, NumDims.x_model_22_Mul_1_out);
Vars.x_model_22_Unsque_8 = reshape(Vars.x_model_22_Mul_1_out, shape);

% Add:
Vars.x_model_22_Add_3_out = Vars.x_model_22_Range_2_o + Vars.x_model_22_Consta_4;
NumDims.x_model_22_Add_3_out = max(NumDims.x_model_22_Range_2_o, NumDims.x_model_22_Consta_4);

% Add:
Vars.x_model_22_Add_4_out = Vars.x_model_22_Range_3_o + Vars.x_model_22_Consta_4;
NumDims.x_model_22_Add_4_out = max(NumDims.x_model_22_Range_3_o, NumDims.x_model_22_Consta_4);

% Concat:
[Vars.x_model_22_Concat_16, NumDims.x_model_22_Concat_16] = onnxConcat(0, {Vars.x_model_22_Unsque_8, Vars.x_model_22_Consta_8}, [NumDims.x_model_22_Unsque_8, NumDims.x_model_22_Consta_8]);

% ConstantOfShape:
[Vars.x_model_22_ConstantO, NumDims.x_model_22_ConstantO] = onnxConstantOfShape(Vars.ConstantOfShapeValue1050, Vars.x_model_22_Concat_16);

% Shape:
[Vars.x_model_22_Shape_7_o, NumDims.x_model_22_Shape_7_o] = onnxShape(Vars.x_model_22_Add_4_out, NumDims.x_model_22_Add_4_out);

% Shape:
[Vars.x_model_22_Shape_8_o, NumDims.x_model_22_Shape_8_o] = onnxShape(Vars.x_model_22_Add_3_out, NumDims.x_model_22_Add_3_out);

% Reshape:
[shape, NumDims.x_model_22_Reshape_1] = prepareReshapeArgs(Vars.x_model_22_Add_4_out, Vars.x_model_22_Concat_8_, NumDims.x_model_22_Add_4_out, 0);
Vars.x_model_22_Reshape_1 = reshape(Vars.x_model_22_Add_4_out, shape{:});

% Reshape:
[shape, NumDims.x_model_22_Reshap_1] = prepareReshapeArgs(Vars.x_model_22_Add_3_out, Vars.x_model_22_Concat_9_, NumDims.x_model_22_Add_3_out, 0);
Vars.x_model_22_Reshap_1 = reshape(Vars.x_model_22_Add_3_out, shape{:});

% Add:
Vars.x_model_22_Add_5_out = Vars.x_model_22_ConstantO + Vars.x_model_22_Squeeze_1;
NumDims.x_model_22_Add_5_out = max(NumDims.x_model_22_ConstantO, NumDims.x_model_22_Squeeze_1);

% Concat:
[Vars.x_model_22_Concat_12, NumDims.x_model_22_Concat_12] = onnxConcat(0, {Vars.x_model_22_Shape_7_o, Vars.x_model_22_Shape_8_o}, [NumDims.x_model_22_Shape_7_o, NumDims.x_model_22_Shape_8_o]);

% Expand:
[shape, NumDims.x_model_22_Expand_2_] = prepareExpandArgs(Vars.x_model_22_Concat_12);
Vars.x_model_22_Expand_2_ = Vars.x_model_22_Reshape_1 + zeros(shape);

% Expand:
[shape, NumDims.x_model_22_Expand_3_] = prepareExpandArgs(Vars.x_model_22_Concat_12);
Vars.x_model_22_Expand_3_ = Vars.x_model_22_Reshap_1 + zeros(shape);

% Unsqueeze:
[shape, NumDims.x_model_22_Unsque_6] = prepareUnsqueezeArgs(Vars.x_model_22_Expand_3_, Vars.x_model_22_Consta_6, NumDims.x_model_22_Expand_3_);
Vars.x_model_22_Unsque_6 = reshape(Vars.x_model_22_Expand_3_, shape);

% Unsqueeze:
[shape, NumDims.x_model_22_Unsque_7] = prepareUnsqueezeArgs(Vars.x_model_22_Expand_2_, Vars.x_model_22_Consta_6, NumDims.x_model_22_Expand_2_);
Vars.x_model_22_Unsque_7 = reshape(Vars.x_model_22_Expand_2_, shape);

% Concat:
[Vars.x_model_22_Concat_15, NumDims.x_model_22_Concat_15] = onnxConcat(-1, {Vars.x_model_22_Unsque_6, Vars.x_model_22_Unsque_7}, [NumDims.x_model_22_Unsque_6, NumDims.x_model_22_Unsque_7]);

% Reshape:
[shape, NumDims.x_model_22_Reshap_2] = prepareReshapeArgs(Vars.x_model_22_Concat_15, Vars.x_model_22_Consta_7, NumDims.x_model_22_Concat_15, 0);
Vars.x_model_22_Reshap_2 = reshape(Vars.x_model_22_Concat_15, shape{:});

% Reshape:
[shape, NumDims.x_model_22_Reshape_2] = prepareReshapeArgs(Vars.x_model_22_Concat_2_, Vars.x_model_22_Concat_3_, NumDims.x_model_22_Concat_2_, 0);
Vars.x_model_22_Reshape_2 = reshape(Vars.x_model_22_Concat_2_, shape{:});

% Shape:
[Vars.x_model_22_Shape_9_o, NumDims.x_model_22_Shape_9_o] = onnxShape(Vars.x_model_22_Concat_2_, NumDims.x_model_22_Concat_2_);

% Concat:
[Vars.x_model_22_Concat_6_, NumDims.x_model_22_Concat_6_] = onnxConcat(2, {Vars.x_model_22_Reshape_o, Vars.x_model_22_Reshap_6, Vars.x_model_22_Reshape_2}, [NumDims.x_model_22_Reshape_o, NumDims.x_model_22_Reshap_6, NumDims.x_model_22_Reshape_2]);

% Gather:
[Vars.x_model_22_Gather_5_, NumDims.x_model_22_Gather_5_] = onnxGather(Vars.x_model_22_Shape_9_o, Vars.x_model_22_Constant_, 0, NumDims.x_model_22_Shape_9_o, NumDims.x_model_22_Constant_);

% Gather:
[Vars.x_model_22_Gather_6_, NumDims.x_model_22_Gather_6_] = onnxGather(Vars.x_model_22_Shape_9_o, Vars.x_model_22_Consta_1, 0, NumDims.x_model_22_Shape_9_o, NumDims.x_model_22_Consta_1);

% Cast:
Vars.x_model_22_Cast_4_ou = single(Vars.x_model_22_Gather_6_);
NumDims.x_model_22_Cast_4_ou = NumDims.x_model_22_Gather_6_;

% Cast:
Vars.x_model_22_Cast_5_ou = single(Vars.x_model_22_Gather_5_);
NumDims.x_model_22_Cast_5_ou = NumDims.x_model_22_Gather_5_;

% Mul:
Vars.x_model_22_Mul_2_out = Vars.x_model_22_Gather_5_ .* Vars.x_model_22_Gather_6_;
NumDims.x_model_22_Mul_2_out = max(NumDims.x_model_22_Gather_5_, NumDims.x_model_22_Gather_6_);

% Split:
[Vars.x_model_22_Split_1_o, Vars.x_model_22_Split_1_1, NumDims.x_model_22_Split_1_o, NumDims.x_model_22_Split_1_1] = onnxSplit13(Vars.x_model_22_Concat_6_, 1, Vars.onnx__Split_580, 2, NumDims.x_model_22_Concat_6_);

% Range:
Vars.x_model_22_Range_4_o = dlarray(Vars.x_model_22_Consta_2:Vars.x_model_22_Consta_3:Vars.x_model_22_Cast_4_ou-sign(Vars.x_model_22_Consta_3))';
NumDims.x_model_22_Range_4_o = 1;

% Range:
Vars.x_model_22_Range_5_o = dlarray(Vars.x_model_22_Consta_2:Vars.x_model_22_Consta_3:Vars.x_model_22_Cast_5_ou-sign(Vars.x_model_22_Consta_3))';
NumDims.x_model_22_Range_5_o = 1;

% Unsqueeze:
[shape, NumDims.x_model_22_Unsque_1] = prepareUnsqueezeArgs(Vars.x_model_22_Mul_2_out, Vars.onnx__Unsqueeze_396, NumDims.x_model_22_Mul_2_out);
Vars.x_model_22_Unsque_1 = reshape(Vars.x_model_22_Mul_2_out, shape);

% Shape:
[Vars.x_model_22_dfl_Shape, NumDims.x_model_22_dfl_Shape] = onnxShape(Vars.x_model_22_Split_1_o, NumDims.x_model_22_Split_1_o);

% Sigmoid:
Vars.x_model_22_Sigmoid_o = sigmoid(Vars.x_model_22_Split_1_1);
NumDims.x_model_22_Sigmoid_o = NumDims.x_model_22_Split_1_1;

% Add:
Vars.x_model_22_Add_6_out = Vars.x_model_22_Range_4_o + Vars.x_model_22_Consta_4;
NumDims.x_model_22_Add_6_out = max(NumDims.x_model_22_Range_4_o, NumDims.x_model_22_Consta_4);

% Add:
Vars.x_model_22_Add_7_out = Vars.x_model_22_Range_5_o + Vars.x_model_22_Consta_4;
NumDims.x_model_22_Add_7_out = max(NumDims.x_model_22_Range_5_o, NumDims.x_model_22_Consta_4);

% Concat:
[Vars.x_model_22_Concat_21, NumDims.x_model_22_Concat_21] = onnxConcat(0, {Vars.x_model_22_Unsque_1, Vars.x_model_22_Consta_8}, [NumDims.x_model_22_Unsque_1, NumDims.x_model_22_Consta_8]);

% Gather:
[Vars.x_model_22_dfl_Gat_1, NumDims.x_model_22_dfl_Gat_1] = onnxGather(Vars.x_model_22_dfl_Shape, Vars.x_model_22_Consta_12, 0, NumDims.x_model_22_dfl_Shape, NumDims.x_model_22_Consta_12);

% Gather:
[Vars.x_model_22_dfl_Gathe, NumDims.x_model_22_dfl_Gathe] = onnxGather(Vars.x_model_22_dfl_Shape, Vars.x_model_22_Constant_, 0, NumDims.x_model_22_dfl_Shape, NumDims.x_model_22_Constant_);

% ConstantOfShape:
[Vars.x_model_22_Constan_1, NumDims.x_model_22_Constan_1] = onnxConstantOfShape(Vars.ConstantOfShapeValue1051, Vars.x_model_22_Concat_21);

% Unsqueeze:
[shape, NumDims.x_model_22_dfl_Uns_1] = prepareUnsqueezeArgs(Vars.x_model_22_dfl_Gat_1, Vars.onnx__Unsqueeze_396, NumDims.x_model_22_dfl_Gat_1);
Vars.x_model_22_dfl_Uns_1 = reshape(Vars.x_model_22_dfl_Gat_1, shape);

% Unsqueeze:
[shape, NumDims.x_model_22_dfl_Unsqu] = prepareUnsqueezeArgs(Vars.x_model_22_dfl_Gathe, Vars.onnx__Unsqueeze_396, NumDims.x_model_22_dfl_Gathe);
Vars.x_model_22_dfl_Unsqu = reshape(Vars.x_model_22_dfl_Gathe, shape);

% Shape:
[Vars.x_model_22_Shape_11_, NumDims.x_model_22_Shape_11_] = onnxShape(Vars.x_model_22_Add_7_out, NumDims.x_model_22_Add_7_out);

% Shape:
[Vars.x_model_22_Shape_12_, NumDims.x_model_22_Shape_12_] = onnxShape(Vars.x_model_22_Add_6_out, NumDims.x_model_22_Add_6_out);

% Reshape:
[shape, NumDims.x_model_22_Reshap_3] = prepareReshapeArgs(Vars.x_model_22_Add_7_out, Vars.x_model_22_Concat_8_, NumDims.x_model_22_Add_7_out, 0);
Vars.x_model_22_Reshap_3 = reshape(Vars.x_model_22_Add_7_out, shape{:});

% Reshape:
[shape, NumDims.x_model_22_Reshap_4] = prepareReshapeArgs(Vars.x_model_22_Add_6_out, Vars.x_model_22_Concat_9_, NumDims.x_model_22_Add_6_out, 0);
Vars.x_model_22_Reshap_4 = reshape(Vars.x_model_22_Add_6_out, shape{:});

% Add:
Vars.x_model_22_Add_8_out = Vars.x_model_22_Constan_1 + Vars.x_model_22_Squeeze_2;
NumDims.x_model_22_Add_8_out = max(NumDims.x_model_22_Constan_1, NumDims.x_model_22_Squeeze_2);

% Concat:
[Vars.x_model_22_dfl_Con_1, NumDims.x_model_22_dfl_Con_1] = onnxConcat(0, {Vars.x_model_22_dfl_Uns_1, Vars.x_model_22_dfl_Const, Vars.x_model_22_dfl_Con_4, Vars.x_model_22_dfl_Unsqu}, [NumDims.x_model_22_dfl_Uns_1, NumDims.x_model_22_dfl_Const, NumDims.x_model_22_dfl_Con_4, NumDims.x_model_22_dfl_Unsqu]);

% Concat:
[Vars.x_model_22_dfl_Conca, NumDims.x_model_22_dfl_Conca] = onnxConcat(0, {Vars.x_model_22_dfl_Uns_1, Vars.x_model_22_dfl_Const, Vars.x_model_22_dfl_Unsqu}, [NumDims.x_model_22_dfl_Uns_1, NumDims.x_model_22_dfl_Const, NumDims.x_model_22_dfl_Unsqu]);

% Concat:
[Vars.x_model_22_Concat_17, NumDims.x_model_22_Concat_17] = onnxConcat(0, {Vars.x_model_22_Shape_11_, Vars.x_model_22_Shape_12_}, [NumDims.x_model_22_Shape_11_, NumDims.x_model_22_Shape_12_]);

% Concat:
[Vars.x_model_22_Concat_23, NumDims.x_model_22_Concat_23] = onnxConcat(0, {Vars.x_model_22_Add_2_out, Vars.x_model_22_Add_5_out, Vars.x_model_22_Add_8_out}, [NumDims.x_model_22_Add_2_out, NumDims.x_model_22_Add_5_out, NumDims.x_model_22_Add_8_out]);

% Reshape:
[shape, NumDims.x_model_22_dfl_Res_1] = prepareReshapeArgs(Vars.x_model_22_Split_1_o, Vars.x_model_22_dfl_Con_1, NumDims.x_model_22_Split_1_o, 0);
Vars.x_model_22_dfl_Res_1 = reshape(Vars.x_model_22_Split_1_o, shape{:});

% Expand:
[shape, NumDims.x_model_22_Expand_4_] = prepareExpandArgs(Vars.x_model_22_Concat_17);
Vars.x_model_22_Expand_4_ = Vars.x_model_22_Reshap_3 + zeros(shape);

% Expand:
[shape, NumDims.x_model_22_Expand_5_] = prepareExpandArgs(Vars.x_model_22_Concat_17);
Vars.x_model_22_Expand_5_ = Vars.x_model_22_Reshap_4 + zeros(shape);

% Transpose:
[perm, NumDims.x_model_22_Transpose] = prepareTransposeArgs(Vars.TransposePerm1052, NumDims.x_model_22_Concat_23);
if ~isempty(perm)
    Vars.x_model_22_Transpose = permute(Vars.x_model_22_Concat_23, perm);
end

% Transpose:
[perm, NumDims.x_model_22_dfl_Trans] = prepareTransposeArgs(Vars.TransposePerm1053, NumDims.x_model_22_dfl_Res_1);
if ~isempty(perm)
    Vars.x_model_22_dfl_Trans = permute(Vars.x_model_22_dfl_Res_1, perm);
end

% Unsqueeze:
[shape, NumDims.x_model_22_Unsque_9] = prepareUnsqueezeArgs(Vars.x_model_22_Expand_5_, Vars.x_model_22_Consta_6, NumDims.x_model_22_Expand_5_);
Vars.x_model_22_Unsque_9 = reshape(Vars.x_model_22_Expand_5_, shape);

% Unsqueeze:
[shape, NumDims.x_model_22_Unsqueeze] = prepareUnsqueezeArgs(Vars.x_model_22_Expand_4_, Vars.x_model_22_Consta_6, NumDims.x_model_22_Expand_4_);
Vars.x_model_22_Unsqueeze = reshape(Vars.x_model_22_Expand_4_, shape);

% Concat:
[Vars.x_model_22_Concat_20, NumDims.x_model_22_Concat_20] = onnxConcat(-1, {Vars.x_model_22_Unsque_9, Vars.x_model_22_Unsqueeze}, [NumDims.x_model_22_Unsque_9, NumDims.x_model_22_Unsqueeze]);

% Reshape:
[shape, NumDims.x_model_22_Reshap_5] = prepareReshapeArgs(Vars.x_model_22_Concat_20, Vars.x_model_22_Consta_7, NumDims.x_model_22_Concat_20, 0);
Vars.x_model_22_Reshap_5 = reshape(Vars.x_model_22_Concat_20, shape{:});

% Concat:
[Vars.x_model_22_Concat_22, NumDims.x_model_22_Concat_22] = onnxConcat(0, {Vars.x_model_22_Reshape_7, Vars.x_model_22_Reshap_2, Vars.x_model_22_Reshap_5}, [NumDims.x_model_22_Reshape_7, NumDims.x_model_22_Reshap_2, NumDims.x_model_22_Reshap_5]);

% Transpose:
[perm, NumDims.x_model_22_Transpo_1] = prepareTransposeArgs(Vars.TransposePerm1054, NumDims.x_model_22_Concat_22);
if ~isempty(perm)
    Vars.x_model_22_Transpo_1 = permute(Vars.x_model_22_Concat_22, perm);
end

% Unsqueeze:
[shape, NumDims.x_model_22_Unsque_2] = prepareUnsqueezeArgs(Vars.x_model_22_Transpo_1, Vars.onnx__Unsqueeze_396, NumDims.x_model_22_Transpo_1);
Vars.x_model_22_Unsque_2 = reshape(Vars.x_model_22_Transpo_1, shape);

% Set graph output arguments from Vars and NumDims:
x_model_22_dfl_Trans = Vars.x_model_22_dfl_Trans;
x_model_22_dfl_TransNumDims1058 = NumDims.x_model_22_dfl_Trans;
x_model_22_dfl_Conca = Vars.x_model_22_dfl_Conca;
x_model_22_dfl_ConcaNumDims1059 = NumDims.x_model_22_dfl_Conca;
x_model_22_Unsque_2 = Vars.x_model_22_Unsque_2;
x_model_22_Unsque_2NumDims1060 = NumDims.x_model_22_Unsque_2;
x_model_22_Transpose = Vars.x_model_22_Transpose;
x_model_22_TransposeNumDims1061 = NumDims.x_model_22_Transpose;
x_model_22_Sigmoid_o = Vars.x_model_22_Sigmoid_o;
x_model_22_Sigmoid_oNumDims1062 = NumDims.x_model_22_Sigmoid_o;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, numDataOutputs, params, varargin)
% Function to validate inputs to Shape_To_TransposeFcn:
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
[inputDataPerms, outputDataPerms, Training] = parseInputs(x_model_22_Concat_ou, x_model_22_Concat_1_, x_model_22_Concat_2_, 5, params, varargin{:});
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

function [x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o] = postprocessOutput(x_model_22_dfl_Trans, x_model_22_dfl_Conca, x_model_22_Unsque_2, x_model_22_Transpose, x_model_22_Sigmoid_o, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(x_model_22_dfl_Trans)
        x_model_22_dfl_Trans = extractdata(x_model_22_dfl_Trans);
    end
    if isdlarray(x_model_22_dfl_Conca)
        x_model_22_dfl_Conca = extractdata(x_model_22_dfl_Conca);
    end
    if isdlarray(x_model_22_Unsque_2)
        x_model_22_Unsque_2 = extractdata(x_model_22_Unsque_2);
    end
    if isdlarray(x_model_22_Transpose)
        x_model_22_Transpose = extractdata(x_model_22_Transpose);
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

function [Y, numDimsY] = onnxConstantOfShape(value, ONNXShape)
% Returns a DLT tensor with the reverse of the ONNXShape.
DLTShape = fliplr(extractdata(ONNXShape(:)'));
numDimsY = numel(DLTShape);
switch numDimsY
    case 0
        % If shape is empty, output is a scalar
        Y = value;
    case 1
        Y = ones(DLTShape,1) .* value;
    otherwise
        Y = ones(DLTShape) .* value;
end
end

function [Y, numDimsY] = onnxGather(X, ONNXIdx, ONNXAxis, numDimsX, numDimsIdx)
% Function implementing the ONNX Gather operator

% In ONNX, 'Gather' first indexes into dimension ONNXAxis of data, using
% the contents of ONNXIdx as the indices. Then, it reshapes the ONNXAxis
% into the shape of ONNXIdx.
%   Example 1:
% Suppose data has shape [2 3 4 5], ONNXIdx has shape [6 7], and axis=1.
% The result has shape [2 6 7 4 5].
%   Example 2:
% Suppose data has shape [2 3 4 5], ONNXIdx has shape [6], and axis=1.
% The result has shape [2 6 4 5].
%   Example 3:
% Suppose data has shape [2 3 4 5], ONNXIdx has shape [] (a scalar), and axis=1.
% The result has shape [2 4 5].
%
% Since we're using reverse indexing relative to ONNX, in this function
% data and ONNXIdx both have reversed dimension ordering.
numDimsY = numDimsIdx + (numDimsX - 1);
if isempty(X)
    Y = X;
    return;
end
% (1) First, do the subsref part of Gather
if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;                                 % Axis can be negative. Convert it to its positive equivalent.
end
dltAxis = numDimsX - ONNXAxis;                                      % Convert axis to DLT. ONNXAxis is origin 0 and we index from the end
ONNXIdx(ONNXIdx<0) = ONNXIdx(ONNXIdx<0) + size(X, dltAxis);         % ONNXIdx can have negative components. Make them positive.
dltIdx  = extractdata(ONNXIdx) + 1;                                 % ONNXIdx is origin-0 in ONNX, so add 1 to get dltIdx
% Use subsref to index into data
Indices.subs = repmat({':'}, 1, numDimsX);
Indices.subs{dltAxis} = dltIdx(:);                                  % Index as a column to ensure the output is 1-D in the indexed dimension (for now).
Indices.type = '()';
Y = subsref(X, Indices);
% (2) Now do the reshaping part of Gather
shape = size(Y, 1:numDimsX);
if numDimsIdx == 0
    % Delete the indexed dimension
    shape(dltAxis) = [];
elseif numDimsIdx > 1
    % Reshape the indexed dimension into the shape of ONNXIdx
    shape = [shape(1:dltAxis-1) size(ONNXIdx, 1:numDimsIdx) shape(dltAxis+1:end)];
end
% Extend the shape to 2D so it's valid MATLAB
if numel(shape) < 2
    shape = [shape ones(1,2-numel(shape))];
end
Y = reshape(Y, shape);
end

function [Y, numDimsY] = onnxShape(X, numDimsX)
% Implements the ONNX Shape operator
% Return the reverse ONNX shape as a 1D column vector
switch numDimsX
    case 0
        if isempty(X)
            Y = dlarray(0);
        else
            Y = dlarray(1);
        end
    case 1
        if isempty(X)
            Y = dlarray(0);
        else
            Y = dlarray(size(X,1));
        end
    otherwise
        Y = dlarray(fliplr(size(X, 1:numDimsX))');
end
numDimsY = 1;
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

function [shape, numDimsY] = prepareExpandArgs(ONNXShape)
% Prepares arguments for implementing the ONNX Expand operator

% Broadcast X to ONNXShape. The shape of X must be compatible with ONNXShape.
ONNXShape = extractdata(ONNXShape);
shape = fliplr(ONNXShape(:)');
if numel(shape) < 2
    shape = [shape ones(1, 2-numel(shape))];
end
numDimsY = numel(ONNXShape);
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

function [newShape, numDimsY] = prepareUnsqueezeArgs(X, ONNXAxes, numDimsX)
% Prepares arguments for implementing the ONNX Unsqueeze operator
numDimsY = numDimsX + numel(ONNXAxes);
ONNXAxes = extractdata(ONNXAxes);
ONNXAxes(ONNXAxes<0) = ONNXAxes(ONNXAxes<0) + numDimsY;
ONNXAxes = sort(ONNXAxes);                                              % increasing order
if numDimsY == 1
    newShape = size(X);
else
    DLTAxes  = flip(numDimsY - ONNXAxes);                                  % increasing order
    newShape = ones(1, numDimsY);
    posToSet = setdiff(1:numDimsY, DLTAxes, 'stable');
    newShape(posToSet) = size(X, 1:numel(posToSet));
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
