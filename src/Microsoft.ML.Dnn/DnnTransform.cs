﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Google.Protobuf;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Dnn;
using NumSharp;
using Tensorflow;
using Tensorflow.Summaries;
using static Microsoft.ML.Transforms.Dnn.DnnUtils;
using static Tensorflow.Python;

[assembly: LoadableClass(DnnTransformer.Summary, typeof(IDataTransform), typeof(DnnTransformer),
    typeof(DnnEstimator.Options), typeof(SignatureDataTransform), DnnTransformer.UserName, DnnTransformer.ShortName)]

[assembly: LoadableClass(DnnTransformer.Summary, typeof(IDataTransform), typeof(DnnTransformer), null, typeof(SignatureLoadDataTransform),
    DnnTransformer.UserName, DnnTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(DnnTransformer), null, typeof(SignatureLoadModel),
    DnnTransformer.UserName, DnnTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(DnnTransformer), null, typeof(SignatureLoadRowMapper),
    DnnTransformer.UserName, DnnTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// <see cref="ITransformer" /> for the <see cref="DnnEstimator"/>.
    /// </summary>
    public sealed class DnnTransformer : RowToRowTransformerBase
    {
        private readonly IHostEnvironment Env;
        private readonly string _savedModelPath;
        private readonly bool _isTemporarySavedModel;
        private readonly bool _addBatchDimensionInput;
        internal Session Session;
        internal readonly Runner Runner;
        internal readonly DataViewType[] OutputTypes;
        internal readonly TF_DataType[] TFOutputTypes;
        internal readonly TF_DataType[] TFInputTypes;
        internal readonly TensorShape[] TFInputShapes;
        internal readonly (Operation, int)[] TFInputOperations;
        internal readonly (Operation, int)[] TFOutputOperations;
        internal Dictionary<string, Operation> OperationCache;
        internal TF_Output[] TFInputNodes;
        internal TF_Output[] TFOutputNodes;
        internal IntPtr[] TFOperations;
        internal Tensor BottleneckTensor;
        internal Operation TrainStep;
        internal Tensor SoftMaxTensor;
        internal Tensor CrossEntropy;
        internal Tensor LabelTensor;
        internal Tensor EvaluationStep;
        internal Tensor Prediction;
        internal readonly int ClassCount;
        internal readonly string CheckpointPath;
        internal DataViewSchema.Column LabelColumn;
        internal readonly string BottleneckOperationName;
        internal Graph Graph => Session.graph;
        internal readonly Dictionary<string, string> IdvToTfMapping;
        internal readonly string[] Inputs;
        internal readonly string[] Outputs;
        internal readonly string[] NonTFOutputs;

        internal static int BatchSize = 1;
        internal const string Summary = "Trains Dnn models.";
        internal const string UserName = "DnnTransform";
        internal const string ShortName = "DnnTransform";
        internal const string LoaderSignature = "DnnTransform";

        internal static class DefaultModelFileNames
        {
            public const string VariablesFolder = "variables";
            public const string Index = "variables.index";
            public const string Data = "variables.data-00000-of-00001";
            public const string Graph = "saved_model.pb";
            public const string TmpMlnetModel = "mlnet_model";
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DNN",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00000001,
                verReadableCur: 0x00000001,
                verWeCanReadBack: 0x00000001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DnnTransformer).Assembly.FullName);
        }

        // Factory method for SignatureLoadModel.
        private static DnnTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // byte: indicator for frozen models
            // byte: indicator for adding batch dimension in input
            // stream: tensorFlow model.
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            GetModelInfo(env, ctx, out string[] inputs, out string[] outputs, out bool isFrozen, out bool addBatchDimensionInput);
            if (isFrozen)
            {
                byte[] modelBytes = null;
                if (!ctx.TryLoadBinaryStream("TFModel", r => modelBytes = r.ReadByteArray()))
                    throw env.ExceptDecode();
                return new DnnTransformer(env, DnnUtils.LoadTFSession(env, modelBytes), outputs, inputs, null, false, addBatchDimensionInput);
            }

            var tempDirPath = Path.GetFullPath(Path.Combine(Path.GetTempPath(), nameof(DnnTransformer) + "_" + Guid.NewGuid()));
            DnnUtils.CreateFolderWithAclIfNotExists(env, tempDirPath);
            try
            {
                var load = ctx.TryLoadBinaryStream("TFSavedModel", br =>
                {
                    int count = br.ReadInt32();
                    for (int n = 0; n < count; n++)
                    {
                        string relativeFile = br.ReadString();
                        long fileLength = br.ReadInt64();

                        string fullFilePath = Path.Combine(tempDirPath, relativeFile);
                        string fullFileDir = Path.GetDirectoryName(fullFilePath);
                        if (fullFileDir != tempDirPath)
                        {
                            DnnUtils.CreateFolderWithAclIfNotExists(env, fullFileDir);
                        }
                        using (var fs = new FileStream(fullFilePath, FileMode.Create, FileAccess.Write))
                        {
                            long actualRead = br.BaseStream.CopyRange(fs, fileLength);
                            env.Assert(actualRead == fileLength);
                        }
                    }
                });

                return new DnnTransformer(env, DnnUtils.GetSession(env, tempDirPath), outputs, inputs, tempDirPath, true, addBatchDimensionInput);
            }
            catch (Exception)
            {
                DnnUtils.DeleteFolderWithRetries(env, tempDirPath);
                throw;
            }
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, DnnEstimator.Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.InputColumns, nameof(options.InputColumns));
            env.CheckValue(options.OutputColumns, nameof(options.OutputColumns));

            return new DnnTransformer(env, options, input).MakeDataTransform(input);
        }

        internal DnnTransformer(IHostEnvironment env, DnnEstimator.Options options, IDataView input)
            : this(env, options, DnnUtils.LoadDnnModel(env, options.ModelLocation), input)
        {
        }

        internal DnnTransformer(IHostEnvironment env, DnnEstimator.Options options, DnnModel tensorFlowModel, IDataView input, IDataView validationSet = null)
            : this(env, tensorFlowModel.Session, options.OutputColumns, options.InputColumns,
                  DnnUtils.IsSavedModel(env, options.ModelLocation) ? options.ModelLocation : null,
                  false, options.AddBatchDimensionInputs, options.BatchSize, options, input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            if (options.ReTrain || options.TransferLearning)
            {
                env.CheckValue(input, nameof(input));

                if (options.ReTrain)
                    CheckTrainingParameters(options);

                if (options.ReTrain && !DnnUtils.IsSavedModel(env, options.ModelLocation))
                    throw env.ExceptNotSupp("TensorFlowTransform: Re-Training of TensorFlow model is only supported for un-frozen model.");

                TrainCore(options, input, validationSet);
            }
        }

        private void CheckTrainingParameters(DnnEstimator.Options options)
        {
            Host.CheckNonWhiteSpace(options.LabelColumn, nameof(options.LabelColumn));
            Host.CheckNonWhiteSpace(options.OptimizationOperation, nameof(options.OptimizationOperation));
            if (Session.graph.OperationByName(options.OptimizationOperation) == null)
                throw Host.ExceptParam(nameof(options.OptimizationOperation), $"Optimization operation '{options.OptimizationOperation}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.TensorFlowLabel, nameof(options.TensorFlowLabel));
            if (Session.graph.OperationByName(options.TensorFlowLabel) == null)
                throw Host.ExceptParam(nameof(options.TensorFlowLabel), $"'{options.TensorFlowLabel}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.SaveLocationOperation, nameof(options.SaveLocationOperation));
            if (Session.graph.OperationByName(options.SaveLocationOperation) == null)
                throw Host.ExceptParam(nameof(options.SaveLocationOperation), $"'{options.SaveLocationOperation}' does not exist in the model");

            Host.CheckNonWhiteSpace(options.SaveOperation, nameof(options.SaveOperation));
            if (Session.graph.OperationByName(options.SaveOperation) == null)
                throw Host.ExceptParam(nameof(options.SaveOperation), $"'{options.SaveOperation}' does not exist in the model");

            if (options.LossOperation != null)
            {
                Host.CheckNonWhiteSpace(options.LossOperation, nameof(options.LossOperation));
                if (Session.graph.OperationByName(options.LossOperation) == null)
                    throw Host.ExceptParam(nameof(options.LossOperation), $"'{options.LossOperation}' does not exist in the model");
            }

            if (options.MetricOperation != null)
            {
                Host.CheckNonWhiteSpace(options.MetricOperation, nameof(options.MetricOperation));
                if (Session.graph.OperationByName(options.MetricOperation) == null)
                    throw Host.ExceptParam(nameof(options.MetricOperation), $"'{options.MetricOperation}' does not exist in the model");
            }

            if (options.LearningRateOperation != null)
            {
                Host.CheckNonWhiteSpace(options.LearningRateOperation, nameof(options.LearningRateOperation));
                if (Session.graph.OperationByName(options.LearningRateOperation) == null)
                    throw Host.ExceptParam(nameof(options.LearningRateOperation), $"'{options.LearningRateOperation}' does not exist in the model");
            }
        }

        private (int, bool, TF_DataType, TensorShape) GetTrainingInputInfo(DataViewSchema inputSchema, string columnName, string tfNodeName, int batchSize)
        {
            if (!inputSchema.TryGetColumnIndex(columnName, out int inputColIndex))
                throw Host.Except($"Column {columnName} doesn't exist");

            var type = inputSchema[inputColIndex].Type;
            var isInputVector = type is VectorDataViewType;

            (Operation inputTensor, int index) = GetOperationFromName(tfNodeName, Session);
            var tfInput = new TF_Input(inputTensor, index);
            var tfInputType = inputTensor.OpType == "Placeholder" ? inputTensor.OutputType(index) :
                inputTensor.InputType(index);
            var tfInputShape = ((Tensor)inputTensor).TensorShape;

            if (isInputVector && (tfInputShape == null || (tfInputShape.NDim == 0)))
            {
                var vecType = (VectorDataViewType)type;
                var colTypeDims = new int[vecType.Dimensions.Length + 1];
                colTypeDims[0] = -1;
                for (int indexLocal = 0; indexLocal < vecType.Dimensions.Length; indexLocal += 1)
                    colTypeDims[indexLocal + 1] = vecType.Dimensions[indexLocal];

                tfInputShape = new TensorShape(colTypeDims);
            }
            if (tfInputShape.NDim != -1)
            {
                var newShape = new int[tfInputShape.NDim];
                newShape[0] = tfInputShape[0] == 0 || tfInputShape[0] == -1 ? batchSize : tfInputShape[0];

                for (int j = 1; j < tfInputShape.NDim; j++)
                    newShape[j] = tfInputShape[j];
                tfInputShape = new TensorShape(newShape);
            }

            var expectedType = DnnUtils.Tf2MlNetType(tfInputType);
            var actualType = type.GetItemType().RawType;
            if (type is KeyDataViewType && actualType == typeof(UInt32))
                actualType = typeof(Int64);

            if (actualType != expectedType.RawType)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", columnName, expectedType.ToString(), type.ToString());

            return (inputColIndex, isInputVector, tfInputType, tfInputShape);
        }

        private void TrainCore(DnnEstimator.Options options, IDataView input, IDataView validationSet)
        {
            var inputsForTraining = new string[Inputs.Length + 1];
            var inputColIndices = new int[inputsForTraining.Length];
            var isInputVector = new bool[inputsForTraining.Length];
            var tfInputTypes = new TF_DataType[inputsForTraining.Length];
            var tfInputShapes = new TensorShape[inputsForTraining.Length];

            for (int i = 0; i < Inputs.Length; i++)
                inputsForTraining[i] = IdvToTfMapping[Inputs[i]];

            var inputSchema = input.Schema;
            for (int i = 0; i < inputsForTraining.Length - 1; i++)
                (inputColIndices[i], isInputVector[i], tfInputTypes[i], tfInputShapes[i]) =
                    GetTrainingInputInfo(inputSchema, Inputs[i], inputsForTraining[i], options.BatchSize);

            var index = inputsForTraining.Length - 1;
            if (options.TransferLearning)
                inputsForTraining[index] = LabelTensor.name.Split(':').First();
            else
                inputsForTraining[index] = options.TensorFlowLabel;

            (inputColIndices[index], isInputVector[index], tfInputTypes[index], tfInputShapes[index]) =
                    GetTrainingInputInfo(inputSchema, options.LabelColumn, inputsForTraining[index], options.BatchSize);

            // Create graph inputs.
            Operation labelOp;
            int labelOpIdx;
            if (options.ReTrain)
                (labelOp, labelOpIdx) = GetOperationFromName(options.TensorFlowLabel, Session);
            else
                (labelOp, labelOpIdx) = GetOperationFromName(LabelTensor.name, Session);

            TF_Output[] tfInputs;

            if (options.ReTrain && !string.IsNullOrEmpty(options.LearningRateOperation))
                tfInputs = new TF_Output[TFInputNodes.Length + 2]; //Inputs + Label + Learning Rate.
            else
                tfInputs = new TF_Output[TFInputNodes.Length + 1]; //Inputs + Label.

            Array.Copy(TFInputNodes, tfInputs, TFInputNodes.Length);

            tfInputs[TFInputNodes.Length] = new TF_Output(labelOp, labelOpIdx);

            if (options.ReTrain)
            {
                var lr = GetOperationFromName(options.LearningRateOperation, Session);
                tfInputs[TFInputNodes.Length + 1] = new TF_Output(lr.Item1, lr.Item2);
            }

            // Create graph outputs.
            /*var fetchList = new List<string>();
            if (options.ReTrain)
            {
                if (options.LossOperation != null)
                    fetchList.Add(options.LossOperation);
                if (options.MetricOperation != null)
                    fetchList.Add(options.MetricOperation);
            }

            TF_Output[] tfOutputs = fetchList.Select(x => { var y = GetOperationFromName(x, Session); return new TF_Output(y.Item1, y.Item2); }).ToArray();*/

            // Create graph operations.
            IntPtr[] ops = null;
            if (options.ReTrain && options.OptimizationOperation != null)
                ops = new[] { c_api.TF_GraphOperationByName(Graph, options.OptimizationOperation) };
            else
                ops = new[] { (IntPtr)TrainStep };

            Saver trainSaver = null;
            FileWriter trainWriter = null;
            Tensor merged = null;
            Runner testSetRunner = null;
            Runner validationSetRunner = null;
            if (options.TransferLearning)
            {
                merged = tf.summary.merge_all();
                trainWriter = tf.summary.FileWriter(Path.Combine(Directory.GetCurrentDirectory(), "train"), Session.graph);
                trainSaver = tf.train.Saver();
                trainSaver.save(Session, CheckpointPath);
            }

            // Instantiate the graph.
            Runner runner;// = new Runner(Session, tfInputs, tfOutputs, ops);
            var cols = input.Schema.Where(c => inputColIndices.Contains(c.Index));

            for (int epoch = 0; epoch < options.Epoch; epoch++)
            {
                using (var cursor = input.GetRowCursor(cols))
                {
                    var srcTensorGetters = GetTensorValueGetters(cursor, inputColIndices, isInputVector, tfInputTypes, tfInputShapes);
                    bool isDataLeft = false;
                    using (var ch = Host.Start("Training TensorFlow model..."))
                    using (var pch = Host.StartProgressChannel("TensorFlow training progress..."))
                    {
                        if (options.ReTrain)
                        {
                            float loss = 0;
                            float metric = 0;
                            pch.SetHeader(new ProgressHeader(new[] { "Loss", "Metric" }, new[] { "Epoch" }), (e) => e.SetProgress(0, epoch, options.Epoch));

                            while (cursor.MoveNext())
                            {
                                for (int i = 0; i < inputsForTraining.Length; i++)
                                {
                                    isDataLeft = true;
                                    srcTensorGetters[i].BufferTrainingData();
                                }

                                if (((cursor.Position + 1) % options.BatchSize) == 0)
                                {
                                    isDataLeft = false;
                                    runner = new Runner(Session);

                                    // Add Learning Rate.
                                    if (!string.IsNullOrEmpty(options.LearningRateOperation))
                                        runner.AddInput(options.LearningRateOperation, new Tensor(options.LearningRate));

                                    // Add operations.
                                    if (!string.IsNullOrEmpty(options.OptimizationOperation))
                                        runner.AddOperation(options.OptimizationOperation);

                                    // Add outputs.
                                    if (options.LossOperation != null)
                                        runner.AddOutputs(options.LossOperation);
                                    if (options.MetricOperation != null)
                                        runner.AddOutputs(options.MetricOperation);

                                    var (l, m) = ExecuteGraphAndRetrieveMetrics(inputsForTraining, srcTensorGetters, runner);
                                    loss += l;
                                    metric += m;
                                }
                            }
                            if (isDataLeft)
                            {
                                isDataLeft = false;
                                ch.Warning("Not training on the last batch. The batch size is less than {0}.", options.BatchSize);
                            }
                            pch.Checkpoint(new double?[] { loss, metric });
                        }
                        else
                        {
                            pch.SetHeader(new ProgressHeader(null, new[] { "Epoch" }), (e) => e.SetProgress(0, epoch, options.Epoch));

                            while (cursor.MoveNext())
                            {
                                for (int i = 0; i < inputsForTraining.Length; i++)
                                {
                                    isDataLeft = true;
                                    srcTensorGetters[i].BufferTrainingData();
                                }

                                if (((cursor.Position + 1) % options.BatchSize) == 0)
                                {
                                    isDataLeft = false;
                                    runner = new Runner(Session);

                                    // Add operations.
                                    runner.AddOperation(TrainStep);

                                    // Feed inputs.
                                    for (int i = 0; i < inputsForTraining.Length; i++)
                                        runner.AddInput(inputsForTraining[i], srcTensorGetters[i].GetBufferedBatchTensor());

                                    // Execute the graph.
                                    runner.Run();
                                }
                            }
                            if (isDataLeft)
                            {
                                isDataLeft = false;
                                ch.Warning("Not training on the last batch. The batch size is less than {0}.", options.BatchSize);
                            }
                        }
                    }
                }

                // Measusre accuracy of the model.
                if (options.TransferLearning)
                {
                    // Test on the training set to get accuracy.
                    using (var cursor = input.GetRowCursor(cols))
                    {
                        var srcTensorGetters = GetTensorValueGetters(cursor, inputColIndices, isInputVector, tfInputTypes, tfInputShapes);

                        float accuracy = 0;
                        float crossEntropy = 0;
                        bool isDataLeft = false;
                        int batch = 0;
                        using (var ch = Host.Start("Test TensorFlow model..."))
                        using (var pch = Host.StartProgressChannel("TensorFlow testing progress..."))
                        {
                            pch.SetHeader(new ProgressHeader(new[] { "Accuracy", "Cross Entropy" }, new[] { "Epoch" }), (e) => e.SetProgress(0, epoch, options.Epoch));

                            while (cursor.MoveNext())
                            {
                                for (int i = 0; i < inputColIndices.Length; i++)
                                {
                                    isDataLeft = true;
                                    srcTensorGetters[i].BufferTrainingData();
                                }

                                if (((cursor.Position + 1) % options.BatchSize) == 0)
                                {
                                    isDataLeft = false;
                                    testSetRunner = new Runner(Session);//, tfInputs, new[] { EvaluationStep._as_tf_output(), CrossEntropy._as_tf_output() }, null);
                                    testSetRunner.AddOutputs(EvaluationStep.name);
                                    testSetRunner.AddOutputs(CrossEntropy.name);
                                    var (acc, ce) = ExecuteGraphAndRetrieveMetrics(inputsForTraining, srcTensorGetters, testSetRunner);
                                    accuracy += acc;
                                    crossEntropy += ce;
                                    batch++;
                                }
                            }
                            if (isDataLeft)
                            {
                                isDataLeft = false;
                                ch.Warning("Not training on the last batch. The batch size is less than {0}.", options.BatchSize);
                            }
                            pch.Checkpoint(new double?[] { accuracy / batch, crossEntropy / batch });
                            ch.Info(MessageSensitivity.None, $"Accuracy: {accuracy / batch}, Cross-Entropy: {crossEntropy / batch}");
                        }
                    }

                    // Test on the validation set.
                    if (validationSet != null)
                    {
                        using (var cursor = validationSet.GetRowCursor(cols))
                        {
                            var srcTensorGetters = GetTensorValueGetters(cursor, inputColIndices, isInputVector, tfInputTypes, tfInputShapes);

                            float accuracy = 0;
                            bool isDataLeft = false;
                            int batch = 0;
                            using (var ch = Host.Start("Test TensorFlow model with validation set..."))
                            using (var pch = Host.StartProgressChannel("TensorFlow validation progress..."))
                            {
                                pch.SetHeader(new ProgressHeader(new[] { "Accuracy" }, new[] { "Epoch" }), (e) => e.SetProgress(0, epoch, options.Epoch));

                                while (cursor.MoveNext())
                                {
                                    for (int i = 0; i < inputColIndices.Length; i++)
                                    {
                                        isDataLeft = true;
                                        srcTensorGetters[i].BufferTrainingData();
                                    }

                                    if (((cursor.Position + 1) % options.BatchSize) == 0)
                                    {
                                        isDataLeft = false;
                                        validationSetRunner = new Runner(Session);//, tfInputs, new[] { EvaluationStep._as_tf_output() }, null);
                                        validationSetRunner.AddOutputs(EvaluationStep.name);
                                        var (acc, _) = ExecuteGraphAndRetrieveMetrics(inputsForTraining, srcTensorGetters, validationSetRunner);
                                        accuracy += acc;
                                        batch++;
                                    }
                                }
                                if (isDataLeft)
                                {
                                    isDataLeft = false;
                                    ch.Warning("Not training on the last batch. The batch size is less than {0}.", options.BatchSize);
                                }
                                pch.Checkpoint(new double?[] { accuracy / batch });
                            }
                        }
                    }
                }
            }

            if (options.ReTrain)
                UpdateModelOnDisk(options.ModelLocation, options);
            else
            {
                trainSaver.save(Session, CheckpointPath);
                UpdateTransferLearningModelOnDisk(options, ClassCount);
            }
        }

        private (float loss, float metric) ExecuteGraphAndRetrieveMetrics(
            string[] inputs,
            ITensorValueGetter[] srcTensorGetters,
            Runner runner)
        {
            float loss = 0;
            float metric = 0;
            for (int i = 0; i < inputs.Length; i++)
                runner.AddInput(inputs[i], srcTensorGetters[i].GetBufferedBatchTensor());

            Tensor[] tensor = runner.Run();
            var buffer = tensor[0].Data();
            loss = tensor.Length > 0 && tensor[0] != IntPtr.Zero ? (float)tensor[0].Data<float>()[0] : 0.0f;
            metric = tensor.Length > 1 && tensor[1] != IntPtr.Zero ? (float)tensor[1].Data<float>()[0] : 0.0f;
            return (loss, metric);
        }

        /// <summary>
        /// Updates the model on the disk.
        /// After retraining Session and Graphs are both up-to-date
        /// However model on disk is not which is used to serialzed to ML.Net stream
        /// </summary>
        private void UpdateModelOnDisk(string modelDir, DnnEstimator.Options options)
        {
            try
            {
                // Save the model on disk
                var path = Path.Combine(modelDir, DefaultModelFileNames.TmpMlnetModel);
                //var input = GetOperationFromName(options.SaveLocationOperation, Session);
                var runner = new Runner(Session); //, new[] { new TF_Output(input.Item1, input.Item2) }, null, new[] { c_api.TF_GraphOperationByName(Graph, options.SaveOperation) });

                runner.AddInput(options.SaveLocationOperation, new Tensor(path))
                    .AddOperation(options.SaveOperation)
                    .Run();

                // Preserve original files
                var variablesPath = Path.Combine(modelDir, DefaultModelFileNames.VariablesFolder);
                var archivePath = Path.Combine(variablesPath + "-" + Guid.NewGuid().ToString());
                Directory.CreateDirectory(archivePath);
                foreach (var f in Directory.GetFiles(variablesPath))
                    File.Copy(f, Path.Combine(archivePath, Path.GetFileName(f)));

                string[] modelFilePaths = null;

                // There are two ways parameters are saved depending on
                // either `saver_def = tf.train.Saver().as_saver_def()` was called in Python before `tf.saved_model.simple_save` or not.
                // If `saver_def = tf.train.Saver().as_saver_def()` was called files are saved in top directory.
                // If not then temporary directory is created in current directory which starts with `mlnet_model`
                // and files are saved there.
                var tmpParamDir = Directory.GetDirectories(modelDir, DefaultModelFileNames.TmpMlnetModel + "*");
                if (tmpParamDir != null && tmpParamDir.Length > 0)
                    modelFilePaths = Directory.GetFiles(tmpParamDir[0]);
                else
                    modelFilePaths = Directory.GetFiles(modelDir, DefaultModelFileNames.TmpMlnetModel + "*");

                foreach (var file in modelFilePaths)
                {
                    if (file.EndsWith(".data-00000-of-00001"))
                    {
                        var destination = Path.Combine(variablesPath, DefaultModelFileNames.Data);
                        if (File.Exists(destination))
                            File.Delete(destination);
                        Directory.Move(file, destination);
                    }
                    if (file.EndsWith(".index"))
                    {
                        var destination = Path.Combine(variablesPath, DefaultModelFileNames.Index);
                        if (File.Exists(destination))
                            File.Delete(destination);
                        Directory.Move(file, destination);
                    }
                }

                if (tmpParamDir != null && tmpParamDir.Length > 0)
                    DnnUtils.DeleteFolderWithRetries(Host, tmpParamDir[0]);
            }
            catch (Exception e)
            {
                throw Host.ExceptIO(e, "Error serializing TensorFlow retrained model to disk.");
            }
        }

        private (Session, Tensor, Tensor, Tensor) BuildEvaluationSession(DnnEstimator.Options options, int classCount)
        {
            var evalGraph = DnnUtils.LoadMetaGraph(options.ModelLocation);
            var evalSess = tf.Session(graph: evalGraph);
            Tensor evaluationStep = null;
            Tensor prediction = null;
            Tensor bottleneckTensor = evalGraph.OperationByName(BottleneckOperationName);

            tf_with(evalGraph.as_default(), graph =>
            {
                var (_, _, groundTruthInput, finalTensor) = AddFinalRetrainOps(classCount, options, bottleneckTensor, false);
                tf.train.Saver().restore(evalSess, Path.Combine(Directory.GetCurrentDirectory(), CheckpointPath));
                (evaluationStep, prediction) = AddEvaluationStep(finalTensor, groundTruthInput);
            });

            return (evalSess, LabelTensor, evaluationStep, prediction);
        }

        private (Tensor, Tensor) AddEvaluationStep(Tensor resultTensor, Tensor groundTruthTensor)
        {
            Tensor evaluationStep = null;
            Tensor correctPrediction = null;

            tf_with(tf.name_scope("accuracy"), scope =>
            {
                tf_with(tf.name_scope("correct_prediction"), delegate
                {
                    Prediction = tf.argmax(resultTensor, 1);
                    correctPrediction = tf.equal(Prediction, groundTruthTensor);
                });

                tf_with(tf.name_scope("accuracy"), delegate
                {
                    evaluationStep = tf.reduce_mean(tf.cast(correctPrediction, tf.float32));
                });
            });

            tf.summary.scalar("accuracy", evaluationStep);
            return (evaluationStep, Prediction);
        }

        private void UpdateTransferLearningModelOnDisk(DnnEstimator.Options options, int classCount)
        {
            var (sess, _, _, _) = BuildEvaluationSession(options, classCount);
            var graph = sess.graph;
            var outputGraphDef = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), new string[] { SoftMaxTensor.name.Split(':')[0], Prediction.name.Split(':')[0] });

            string frozenModelPath = CheckpointPath + ".pb";
            File.WriteAllBytes(CheckpointPath + ".pb", outputGraphDef.ToByteArray());
            Session = LoadTFSessionByModelFilePath(Env, frozenModelPath, false);

        }

        private void VariableSummaries(RefVariable var)
        {
            tf_with(tf.name_scope("summaries"), delegate
            {
                var mean = tf.reduce_mean(var);
                tf.summary.scalar("mean", mean);
                Tensor stddev = null;
                tf_with(tf.name_scope("stddev"), delegate
                {
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)));
                });
                tf.summary.scalar("stddev", stddev);
                tf.summary.scalar("max", tf.reduce_max(var));
                tf.summary.scalar("min", tf.reduce_min(var));
                tf.summary.histogram("histogram", var);
            });
        }

        private (Operation, Tensor, Tensor, Tensor) AddFinalRetrainOps(int classCount,
            DnnEstimator.Options options, Tensor bottleneckTensor, bool isTraining)
        {
            var (batch_size, bottleneck_tensor_size) = (bottleneckTensor.TensorShape.Dimensions[0], bottleneckTensor.TensorShape.Dimensions[1]);
            tf_with(tf.name_scope("input"), scope =>
            {
                LabelTensor = tf.placeholder(tf.int64, new TensorShape(batch_size), name: options.LabelColumn);
            });

            string layerName = "final_retrain_ops";
            Tensor logits = null;
            tf_with(tf.name_scope(layerName), scope =>
            {
                RefVariable layerWeights = null;
                tf_with(tf.name_scope("weights"), delegate
                {
                    var initialValue = tf.truncated_normal(new int[] { bottleneck_tensor_size, classCount }, stddev: 0.001f);
                    layerWeights = tf.Variable(initialValue, name: "final_weights");
                    VariableSummaries(layerWeights);
                });

                RefVariable layerBiases = null;
                tf_with(tf.name_scope("biases"), delegate
                {
                    layerBiases = tf.Variable(tf.zeros(classCount), name: "final_biases");
                    VariableSummaries(layerBiases);
                });

                tf_with(tf.name_scope("Wx_plus_b"), delegate
                {
                    logits = tf.matmul(bottleneckTensor, layerWeights) + layerBiases;
                    tf.summary.histogram("pre_activations", logits);
                });
            });

            SoftMaxTensor = tf.nn.softmax(logits, name: options.ScoreColumnName);

            tf.summary.histogram("activations", SoftMaxTensor);
            if (!isTraining)
                return (null, null, LabelTensor, SoftMaxTensor);

            Tensor crossEntropyMean = null;
            tf_with(tf.name_scope("cross_entropy"), delegate
            {
                crossEntropyMean = tf.losses.sparse_softmax_cross_entropy(
                    labels: LabelTensor, logits: logits);
            });

            tf.summary.scalar("cross_entropy", crossEntropyMean);

            tf_with(tf.name_scope("train"), delegate
            {
                var optimizer = tf.train.GradientDescentOptimizer(options.LearningRate);
                TrainStep = optimizer.minimize(crossEntropyMean);
            });

            return (TrainStep, crossEntropyMean, LabelTensor, SoftMaxTensor);
        }

        private void AddTransferLearningLayer(DnnEstimator.Options options, int classCount)
        {
            BottleneckTensor = Graph.OperationByName(BottleneckOperationName);
            tf_with(Graph.as_default(), delegate
            {
                (TrainStep, CrossEntropy, LabelTensor, SoftMaxTensor) =
                    AddFinalRetrainOps(classCount, options, BottleneckTensor, true);
            });
        }

        private static ITensorValueGetter CreateTensorValueGetter<T>(DataViewRow input, bool isVector, int colIndex, TensorShape tfShape, bool keyType = false)
        {
            if (isVector)
                return new TensorValueGetterVec<T>(input, colIndex, tfShape);
            return new TensorValueGetter<T>(input, colIndex, tfShape, keyType);
        }

        private static ITensorValueGetter CreateTensorValueGetter(DataViewRow input, TF_DataType tfType, bool isVector, int colIndex, TensorShape tfShape)
        {
            var type = DnnUtils.Tf2MlNetType(tfType);
            if (input.Schema[colIndex].Type is KeyDataViewType && type.RawType == typeof(Int64))
                return Utils.MarshalInvoke(CreateTensorValueGetter<int>, typeof(UInt32), input, isVector, colIndex, tfShape, true);

            return Utils.MarshalInvoke(CreateTensorValueGetter<int>, type.RawType, input, isVector, colIndex, tfShape, false);
        }

        private static ITensorValueGetter[] GetTensorValueGetters(
            DataViewRow input,
            int[] inputColIndices,
            bool[] isInputVector,
            TF_DataType[] tfInputTypes,
            TensorShape[] tfInputShapes)
        {
            var srcTensorGetters = new ITensorValueGetter[inputColIndices.Length];
            for (int i = 0; i < inputColIndices.Length; i++)
            {
                int colIndex = inputColIndices[i];
                srcTensorGetters[i] = CreateTensorValueGetter(input, tfInputTypes[i], isInputVector[i], colIndex, tfInputShapes[i]);
            }
            return srcTensorGetters;
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private static void GetModelInfo(IHostEnvironment env, ModelLoadContext ctx, out string[] inputs, out string[] outputs, out bool isFrozen, out bool addBatchDimensionInput)
        {
            isFrozen = true;
            bool isNonFrozenModelSupported = ctx.Header.ModelVerReadable >= 0x00010002;
            if (isNonFrozenModelSupported)
                isFrozen = ctx.Reader.ReadBoolByte();

            addBatchDimensionInput = false;
            bool isAddingBatchDimensionSupported = ctx.Header.ModelVerReadable >= 0x00010003;
            if (isAddingBatchDimensionSupported)
                addBatchDimensionInput = ctx.Reader.ReadBoolByte();

            var numInputs = ctx.Reader.ReadInt32();
            env.CheckDecode(numInputs > 0);
            inputs = new string[numInputs];
            for (int j = 0; j < inputs.Length; j++)
                inputs[j] = ctx.LoadNonEmptyString();

            bool isMultiOutput = ctx.Header.ModelVerReadable >= 0x00010002;
            var numOutputs = 1;
            if (isMultiOutput)
                numOutputs = ctx.Reader.ReadInt32();

            env.CheckDecode(numOutputs > 0);
            outputs = new string[numOutputs];
            for (int j = 0; j < outputs.Length; j++)
                outputs[j] = ctx.LoadNonEmptyString();
        }

        internal DnnTransformer(IHostEnvironment env, Session session, string[] outputColumnNames,
            string[] inputColumnNames, string savedModelPath, bool isTemporarySavedModel,
            bool addBatchDimensionInput, int batchSize = 1, DnnEstimator.Options options = null, IDataView input = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(DnnTransformer)))

        {
            Host.CheckValue(session, nameof(session));
            Host.CheckNonEmpty(inputColumnNames, nameof(inputColumnNames));
            Host.CheckNonEmpty(outputColumnNames, nameof(outputColumnNames));

            Env = env;
            Session = session;
            _savedModelPath = savedModelPath;
            _isTemporarySavedModel = isTemporarySavedModel;
            _addBatchDimensionInput = addBatchDimensionInput;
            Inputs = inputColumnNames;
            Outputs = outputColumnNames;
            IdvToTfMapping = new Dictionary<string, string>();

            if (options.TransferLearning)
            {
                var inputSchema = input.Schema;
                var labelColumn = inputSchema.GetColumnOrNull(options.LabelColumn).Value;
                var labelType = labelColumn.Type;
                var labelCount = labelType.GetKeyCount();
                if (labelCount <= 0)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", labelColumn.Name, "Key", labelType.ToString());

                ClassCount = labelCount == 1 ? 2 : (int)labelCount;
                CheckpointPath = Path.Combine(Directory.GetCurrentDirectory(), options.ModelLocation + options.CheckpointName);

                // Configure bottleneck tensor based on the model.
                if (options.Arch == DnnEstimator.Architecture.ResnetV2101)
                    BottleneckOperationName = "resnet_v2_101/SpatialSqueeze";
                else if(options.Arch == DnnEstimator.Architecture.InceptionV3)
                    BottleneckOperationName = "module_apply_default/hub_output/feature_vector/SpatialSqueeze";

                IdvToTfMapping[Inputs[0]] = "input";

                Outputs = new[] { options.ScoreColumnName, options.PredictedLabelColumnName };

                // Add transfer learning layer.
                AddTransferLearningLayer(options, ClassCount);

                // Initialize the variables.
                new Runner(Session).AddOperation(tf.global_variables_initializer()).Run();

                // Add evaluation layer.
                (EvaluationStep, _) = AddEvaluationStep(SoftMaxTensor, LabelTensor);

                IdvToTfMapping[options.ScoreColumnName] = SoftMaxTensor.name;
                IdvToTfMapping[options.PredictedLabelColumnName] = Prediction.name;

                (TFOutputTypes, OutputTypes, TFOutputOperations) = GetOutputInfo(Host, Session, new[] { SoftMaxTensor.name, Prediction.name });
            }
            else
            {
                foreach (var x in Inputs)
                    IdvToTfMapping[x] = x;

                foreach (var x in Outputs)
                    IdvToTfMapping[x] = x;

                (TFOutputTypes, OutputTypes, TFOutputOperations) = GetOutputInfo(Host, Session, Outputs);

            }
            (TFInputTypes, TFInputShapes, TFInputOperations) = GetInputInfo(Host, Session, Inputs.Select(x => IdvToTfMapping[x]).ToArray(), batchSize);

            TFInputNodes = new TF_Output[Inputs.Length];
            TFOutputNodes = new TF_Output[Outputs.Length];

            for (int index = 0; index < TFInputOperations.Length; index += 1)
                TFInputNodes[index] = new TF_Output(TFInputOperations[index].Item1, TFInputOperations[index].Item2);

            for (int index = 0; index < TFOutputOperations.Length; index += 1)
                TFOutputNodes[index] = new TF_Output(TFOutputOperations[index].Item1, TFOutputOperations[index].Item2);
        }

        private static (Operation, int) GetOperationFromName(string operation, Session session)
        {
            var p = operation.IndexOf(':');

            if (p != -1 && p != operation.Length - 1)
            {
                var op = operation.Substring(0, p);
                if (int.TryParse(operation.Substring(p + 1), out var idx))
                {

                    return (session.graph.OperationByName(op), idx);
                }
            }
            return (session.graph.OperationByName(operation), 0);
        }

        internal static (TF_DataType[] tfInputTypes, TensorShape[] tfInputShapes, (Operation, int)[]) GetInputInfo(IHost host, Session session, string[] inputs, int batchSize = 1)
        {
            var tfInputTypes = new TF_DataType[inputs.Length];
            var tfInputShapes = new TensorShape[inputs.Length];
            var tfInputOperations = new (Operation, int)[inputs.Length];

            int index = 0;
            foreach (var input in inputs)
            {
                host.CheckNonWhiteSpace(input, nameof(inputs));
                (Operation inputTensor, int inputTensorIndex) = GetOperationFromName(input, session);

                if (inputTensor == null)
                    throw host.ExceptParam(nameof(inputs), $"Input column '{input}' does not exist in the model");

                TF_DataType tfInputType = string.Compare(inputTensor.OpType, "PlaceHolder", true) == 0 ? inputTensor.OutputType(inputTensorIndex) : inputTensor.InputType(index);
                if (!DnnUtils.IsTypeSupported(tfInputType))
                    throw host.ExceptParam(nameof(session), $"Input type '{tfInputType}' of input column '{input}' is not supported in TensorFlow");

                tfInputTypes[index] = tfInputType;
                tfInputShapes[index] = ((Tensor)inputTensor).TensorShape;
                tfInputOperations[index] = (inputTensor, inputTensorIndex);
                index++;
            }

            return (tfInputTypes, tfInputShapes, tfInputOperations);
        }

        internal static TensorShape GetTensorShape(TF_Output output, Graph graph, Status status = null)
        {
            if (graph == IntPtr.Zero)
                new ObjectDisposedException(nameof(graph));

            var cstatus = status == null ? new Status() : status;
            var n = c_api.TF_GraphGetTensorNumDims(graph, output, cstatus);

            cstatus.Check();

            if (n == -1)
                return new TensorShape(new int[0]);

            var dims = new long[n];
            c_api.TF_GraphGetTensorShape(graph, output, dims, dims.Length, cstatus);
            cstatus.Check();
            return new TensorShape(dims.Select(x => (int)x).ToArray());
        }

        internal static (TF_DataType[] tfOutputTypes, DataViewType[] outputTypes, (Operation, int)[]) GetOutputInfo(IHost host, Session session, string[] outputs)
        {
            var tfOutputTypes = new TF_DataType[outputs.Length];
            var outputTypes = new DataViewType[outputs.Length];
            var newNames = new HashSet<string>();
            var tfOutputOperations = new (Operation, int)[outputs.Length];

            for (int i = 0; i < outputs.Length; i++)
            {
                host.CheckNonWhiteSpace(outputs[i], nameof(outputs));
                if (!newNames.Add(outputs[i]))
                    throw host.ExceptParam(nameof(outputs), $"Output column '{outputs[i]}' specified multiple times");

                (Tensor outputTensor, int outputIndex) = GetOperationFromName(outputs[i], session);
                if (outputTensor == null)
                    throw host.ExceptParam(nameof(outputs), $"Output column '{outputs[i]}' does not exist in the model");

                var tfOutputType = ((Operation)outputTensor).OutputType(outputIndex);
                var shape = GetTensorShape(new TF_Output((Operation)outputTensor, outputIndex), session.graph);

                // The transformer can only retreive the output as fixed length vector with shape of kind [-1, d1, d2, d3, ...]
                // i.e. the first dimension (if unknown) is assumed to be batch dimension.
                // If there are other dimension that are unknown the transformer will return a variable length vector.
                // This is the work around in absence of reshape transformer.
                int[] dims = shape.NDim > 0 ? shape.Dimensions.Skip(shape[0] == -1 ? 1 : 0).ToArray() : new[] { 0 };
                for (int j = 0; j < dims.Length; j++)
                    dims[j] = dims[j] == -1 ? 0 : dims[j];
                if (dims == null || dims.Length == 0)
                {
                    dims = new[] { 1 };
                    outputTypes[i] = DnnUtils.Tf2MlNetType(tfOutputType);
                }
                else
                {
                    var type = DnnUtils.Tf2MlNetType(tfOutputType);
                    outputTypes[i] = new VectorDataViewType(type, dims);
                }

                tfOutputTypes[i] = tfOutputType;
                tfOutputOperations[i] = (outputTensor, outputIndex);
            }

            return (tfOutputTypes, outputTypes, tfOutputOperations);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema) => new Mapper(this, inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // byte: indicator for frozen models
            // byte: indicator for adding batch dimension in input
            // stream: tensorFlow model.
            // int: number of input columns
            // for each input column
            //   int: id of int column name
            // int: number of output columns
            // for each output column
            //   int: id of output column name
            var isFrozen = string.IsNullOrEmpty(_savedModelPath);
            ctx.Writer.WriteBoolByte(isFrozen);
            ctx.Writer.WriteBoolByte(_addBatchDimensionInput);
            if (isFrozen)
            {
                Status status = new Status();
                var buffer = Session.graph.ToGraphDef(status);
                ctx.SaveBinaryStream("TFModel", w =>
                {
                    w.WriteByteArray(buffer.Data);
                });
            }
            else
            {
                ctx.SaveBinaryStream("TFSavedModel", w =>
                {
                    // only these files need to be saved.
                    string[] modelFilePaths =
                    {
                        Path.Combine(_savedModelPath, DefaultModelFileNames.Graph),
                        Path.Combine(_savedModelPath, DefaultModelFileNames.VariablesFolder, DefaultModelFileNames.Data),
                        Path.Combine(_savedModelPath, DefaultModelFileNames.VariablesFolder, DefaultModelFileNames.Index),
                    };

                    w.Write(modelFilePaths.Length);

                    foreach (var fullPath in modelFilePaths)
                    {
                        var relativePath = fullPath.Substring(_savedModelPath.Length + 1);
                        w.Write(relativePath);

                        using (var fs = new FileStream(fullPath, FileMode.Open))
                        {
                            long fileLength = fs.Length;
                            w.Write(fileLength);
                            long actualWritten = fs.CopyRange(w.BaseStream, fileLength);
                            Host.Assert(actualWritten == fileLength);
                        }
                    }
                });
            }
            Host.AssertNonEmpty(Inputs);
            ctx.Writer.Write(Inputs.Length);
            foreach (var colName in Inputs)
                ctx.SaveNonEmptyString(colName);

            Host.AssertNonEmpty(Outputs);
            ctx.Writer.Write(Outputs.Length);
            foreach (var colName in Outputs)
                ctx.SaveNonEmptyString(colName);
        }

        ~DnnTransformer()
        {
            Dispose(false);
        }

        private void Dispose(bool disposing)
        {
            // Ensure that the Session is not null and it's handle is not Zero, as it may have already been disposed/finalized.
            // Technically we shouldn't be calling this if disposing == false, since we're running in finalizer
            // and the GC doesn't guarantee ordering of finalization of managed objects, but we have to make sure
            // that the Session is closed before deleting our temporary directory.
            try
            {
                if (Session != null && Session != IntPtr.Zero)
                {
                    Session.close();
                }
            }
            finally
            {
                if (!string.IsNullOrEmpty(_savedModelPath) && _isTemporarySavedModel)
                {
                    DnnUtils.DeleteFolderWithRetries(Host, _savedModelPath);
                }
            }
        }

        private sealed class Mapper : MapperBase
        {
            private readonly DnnTransformer _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private readonly TensorShape[] _fullySpecifiedShapes;
            private readonly ConcurrentBag<Runner> _runners;

            public Mapper(DnnTransformer parent, DataViewSchema inputSchema) :
                   base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                Host.CheckValue(parent, nameof(parent));
                _parent = parent;
                _inputColIndices = new int[_parent.Inputs.Length];
                _isInputVector = new bool[_parent.Inputs.Length];
                _fullySpecifiedShapes = new TensorShape[_parent.Inputs.Length];
                for (int i = 0; i < _parent.Inputs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.Inputs[i], out _inputColIndices[i]))
                        throw Host.ExceptSchemaMismatch(nameof(InputSchema), "source", _parent.Inputs[i]);

                    var type = inputSchema[_inputColIndices[i]].Type;
                    if (type is VectorDataViewType vecType && vecType.Size == 0)
                        throw Host.Except("Variable length input columns not supported");

                    _isInputVector[i] = type is VectorDataViewType;
                    if (!_isInputVector[i])
                        throw Host.Except("Non-vector columns are not supported and should be loaded as vector columns of size 1");
                    vecType = (VectorDataViewType)type;
                    var expectedType = DnnUtils.Tf2MlNetType(_parent.TFInputTypes[i]);
                    if (type.GetItemType() != expectedType)
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.Inputs[i], expectedType.ToString(), type.ToString());
                    var originalShape = _parent.TFInputShapes[i];
                    var shape = originalShape.Dimensions;

                    var colTypeDims = vecType.Dimensions.Select(dim => (int)dim).ToArray();
                    if (shape == null || (shape.Length == 0))
                        _fullySpecifiedShapes[i] = new TensorShape(colTypeDims);
                    else
                    {
                        // If the column is one dimension we make sure that the total size of the TF shape matches.
                        // Compute the total size of the known dimensions of the shape.
                        int valCount = 1;
                        int numOfUnkDim = 0;
                        foreach (var s in shape)
                        {
                            if (s > 0)
                                valCount *= s;
                            else
                                numOfUnkDim++;
                        }
                        // The column length should be divisible by this, so that the other dimensions can be integral.
                        int typeValueCount = type.GetValueCount();
                        if (typeValueCount % valCount != 0)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {typeValueCount}.");

                        // If the shape is multi-dimensional, we should be able to create the length of the vector by plugging
                        // in a single value for the unknown shapes. For example, if the shape is [?,?,3], then there should exist a value
                        // d such that d*d*3 is equal to the length of the input column.
                        var d = numOfUnkDim > 0 ? Math.Pow(typeValueCount / valCount, 1.0 / numOfUnkDim) : 0;
                        if (d - (int)d != 0)
                            throw Contracts.Except($"Input shape mismatch: Input '{_parent.Inputs[i]}' has shape {originalShape.ToString()}, but input data is of length {typeValueCount}.");

                        // Fill in the unknown dimensions.
                        var l = new int[originalShape.NDim];
                        for (int ishape = 0; ishape < originalShape.NDim; ishape++)
                            l[ishape] = originalShape[ishape] == -1 ? (int)d : originalShape[ishape];
                        _fullySpecifiedShapes[i] = new TensorShape(l);
                    }

                    if (_parent._addBatchDimensionInput)
                    {
                        var l = new int[_fullySpecifiedShapes[i].NDim + 1];
                        l[0] = 1;
                        for (int ishape = 1; ishape < l.Length; ishape++)
                            l[ishape] = _fullySpecifiedShapes[i][ishape - 1];
                        _fullySpecifiedShapes[i] = new TensorShape(l);
                    }
                }

                _runners = new ConcurrentBag<Runner>();
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            private class OutputCache
            {
                public long Position;
                public Dictionary<string, Tensor> Outputs;
                public OutputCache()
                {
                    Position = -1;
                    Outputs = new Dictionary<string, Tensor>();
                }
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.AssertValue(input);

                var outputCache = new OutputCache();
                var activeOutputColNames = _parent.Outputs.Where((x, i) => activeOutput(i)).ToArray();

                var type = DnnUtils.Tf2MlNetType(_parent.TFOutputTypes[iinfo]).RawType;
                Host.Assert(type == _parent.OutputTypes[iinfo].GetItemType().RawType);
                var srcTensorGetters = GetTensorValueGetters(input, _inputColIndices, _isInputVector, _parent.TFInputTypes, _fullySpecifiedShapes);
                return Utils.MarshalInvoke(MakeGetter<int>, type, input, iinfo, srcTensorGetters, activeOutputColNames, outputCache);
            }

            private Delegate MakeGetter<T>(DataViewRow input, int iinfo, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                Host.AssertValue(input);

                if (_parent.OutputTypes[iinfo].IsStandardScalar())
                {
                    ValueGetter<T> valuegetter = (ref T dst) =>
                    {
                        UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                        var tensor = outputCache.Outputs[_parent.Outputs[iinfo]];
                        dst = tensor.Data<T>()[0];
                    };
                    return valuegetter;
                }
                else
                {
                    if (_parent.TFOutputTypes[iinfo] == TF_DataType.TF_STRING)
                    {
                        ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                        {
                            UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                            var tensor = outputCache.Outputs[_parent.Outputs[iinfo]];
                            var tensorSize = tensor.TensorShape.Dimensions.Where(x => x > 0).Aggregate((x, y) => x * y);

                            var editor = VBufferEditor.Create(ref dst, (int)tensorSize);
                            DnnUtils.FetchStringData(tensor, editor.Values);
                            dst = editor.Commit();
                        };
                        return valuegetter;
                    }
                    else
                    {
                        ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                        {
                            UpdateCacheIfNeeded(input.Position, srcTensorGetters, activeOutputColNames, outputCache);

                            var tensor = outputCache.Outputs[_parent.Outputs[iinfo]];
                            var tensorSize = tensor.TensorShape.Dimensions.Where(x => x > 0).Aggregate((x, y) => x * y);

                            var editor = VBufferEditor.Create(ref dst, (int)tensorSize);

                            DnnUtils.FetchData<T>(tensor.Data<T>(), editor.Values);
                            dst = editor.Commit();
                        };
                        return valuegetter;
                    }
                }
            }

            private void UpdateCacheIfNeeded(long position, ITensorValueGetter[] srcTensorGetters, string[] activeOutputColNames, OutputCache outputCache)
            {
                if (outputCache.Position != position)
                {
                    Runner runner = new Runner(_parent.Session);

                    // Feed the inputs.
                    for (int i = 0; i < _parent.Inputs.Length; i++)
                        runner.AddInput(_parent.IdvToTfMapping[_parent.Inputs[i]], srcTensorGetters[i].GetTensor());

                    // Add outputs.
                    for (int i = 0; i < _parent.Outputs.Length; i++)
                        runner.AddOutputs(_parent.IdvToTfMapping[_parent.Outputs[i]]);

                    // Execute the graph.
                    var tensors = runner.Run();
                    Contracts.Assert(tensors.Length > 0);

                    for (int j = 0; j < activeOutputColNames.Length; j++)
                        outputCache.Outputs[activeOutputColNames[j]] = tensors[j];

                    outputCache.Position = position;
                }
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => Enumerable.Range(0, _parent.Outputs.Length).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[_parent.Outputs.Length];
                for (int i = 0; i < _parent.Outputs.Length; i++)
                    info[i] = new DataViewSchema.DetachedColumn(_parent.Outputs[i], _parent.OutputTypes[i], null);
                return info;
            }
        }

        private interface ITensorValueGetter
        {
            Tensor GetTensor();

            void BufferTrainingData();

            Tensor GetBufferedBatchTensor();
        }

        private class TensorValueGetter<T> : ITensorValueGetter
        {
            private readonly ValueGetter<T> _srcgetter;
            private readonly T[] _bufferedData;
            private readonly Int64[] _bufferedDataLong;
            private readonly TensorShape _tfShape;
            private int _position;
            private readonly bool _keyType;

            public TensorValueGetter(DataViewRow input, int colIndex, TensorShape tfShape, bool keyType = false)
            {
                _srcgetter = input.GetGetter<T>(input.Schema[colIndex]);
                _tfShape = tfShape;
                long size = 0;
                _position = 0;
                if (tfShape.Dimensions.Length != 0)
                {
                    size = 1;
                    foreach (var dim in tfShape.Dimensions)
                        size *= dim;
                }
                if (keyType)
                    _bufferedDataLong = new long[size];
                else
                    _bufferedData = new T[size];
                _keyType = keyType;
            }

            public Tensor GetTensor()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                if (_keyType)
                {
                    var tensor = new Tensor(new[] { Convert.ToInt64(scalar) - 1 });
                    tensor.SetShape(_tfShape);
                    return tensor;
                }
                else
                {
                    var tensor = new Tensor(new[] { scalar });
                    tensor.SetShape(_tfShape);
                    return tensor;
                }
            }

            public void BufferTrainingData()
            {
                if (_keyType)
                {
                    var scalar = default(T);
                    _srcgetter(ref scalar);
                    _bufferedDataLong[_position++] = Convert.ToInt64(scalar) - 1;
                }
                else
                {
                    var scalar = default(T);
                    _srcgetter(ref scalar);
                    _bufferedData[_position++] = scalar;
                }
            }

            public Tensor GetBufferedBatchTensor()
            {
                if (_keyType)
                {
                    var tensor = new Tensor(new NDArray(_bufferedDataLong, _tfShape));
                    _position = 0;
                    return tensor;
                }
                else
                {
                    var tensor = new Tensor(new NDArray(_bufferedData, _tfShape));
                    _position = 0;
                    return tensor;
                }
            }
        }

        private class TensorValueGetterVec<T> : ITensorValueGetter
        {
            private readonly ValueGetter<VBuffer<T>> _srcgetter;
            private readonly TensorShape _tfShape;
            private VBuffer<T> _vBuffer;
            private T[] _denseData;
            private T[] _bufferedData;
            private int _position;
            private long[] _dims;
            private readonly long _bufferedDataSize;

            public TensorValueGetterVec(DataViewRow input, int colIndex, TensorShape tfShape)
            {
                _srcgetter = input.GetGetter<VBuffer<T>>(input.Schema[colIndex]);
                _tfShape = tfShape;
                _vBuffer = default;
                _denseData = default;

                long size = 0;
                _position = 0;
                if (tfShape.Dimensions.Length != 0)
                {
                    size = 1;
                    foreach (var dim in tfShape.Dimensions)
                        size *= dim;
                }
                _bufferedData = new T[size];
                _bufferedDataSize = size;
                if (_tfShape.Dimensions != null)
                    _dims = _tfShape.Dimensions.Select(x => (long)x).ToArray();
            }

            public Tensor GetTensor()
            {
                _srcgetter(ref _vBuffer);

                // _denseData.Length can be greater than _vBuffer.Length sometime after
                // Utils.EnsureSize is executed. Use _vBuffer.Length to access the elements in _denseData.
                // This is done to reduce memory allocation every time tensor is created.
                _denseData = new T[_vBuffer.Length];
                _vBuffer.CopyTo(_denseData);
                return CastDataAndReturnAsTensor(_denseData);
            }

            private Tensor CastDataAndReturnAsTensor(T[] data)
            {
                if (typeof(T) == typeof(sbyte))
                    return new Tensor((sbyte[])(object)data, _dims, TF_DataType.TF_INT8);
                else if (typeof(T) == typeof(long))
                    return new Tensor((long[])(object)data, _dims, TF_DataType.TF_INT64);
                else if (typeof(T) == typeof(Int32))
                    return new Tensor((Int32[])(object)data, _dims, TF_DataType.TF_INT32);
                else if (typeof(T) == typeof(Int16))
                    return new Tensor((Int16[])(object)data, _dims, TF_DataType.TF_INT16);
                else if (typeof(T) == typeof(byte))
                    return new Tensor((byte[])(object)data, _dims, TF_DataType.TF_UINT8);
                else if (typeof(T) == typeof(ulong))
                    return new Tensor((ulong[])(object)data, _dims, TF_DataType.TF_UINT64);
                else if (typeof(T) == typeof(UInt32))
                    return new Tensor((UInt32[])(object)data, _dims, TF_DataType.TF_UINT32);
                else if (typeof(T) == typeof(UInt16))
                    return new Tensor((UInt16[])(object)data, _dims, TF_DataType.TF_UINT16);
                else if (typeof(T) == typeof(bool))
                    return new Tensor((bool[])(object)data, _dims, TF_DataType.TF_BOOL);
                else if (typeof(T) == typeof(float))
                    return new Tensor((float[])(object)data, _dims, TF_DataType.TF_FLOAT);
                else if (typeof(T) == typeof(float))
                    return new Tensor((double[])(object)data, _dims, TF_DataType.TF_DOUBLE);
                else if (typeof(T) == typeof(ReadOnlyMemory<char>))
                {
                    byte[][] bytes = new byte[_vBuffer.Length][];
                    for (int i = 0; i < bytes.Length; i++)
                    {
                        bytes[i] = Encoding.UTF8.GetBytes(((ReadOnlyMemory<char>)(object)data[i]).ToArray());
                    }

                    return new Tensor(bytes, _tfShape.dims.Select(x => (long)x).ToArray());
                }

                return new Tensor(new NDArray(data, _tfShape));
            }

            public void BufferTrainingData()
            {
                _srcgetter(ref _vBuffer);
                _vBuffer.CopyTo(_bufferedData, _position);
                _position += _vBuffer.Length;
            }

            public Tensor GetBufferedBatchTensor()
            {
                _position = 0;
                var tensor = CastDataAndReturnAsTensor(_bufferedData);
                _bufferedData = new T[_bufferedDataSize];
                return tensor;
            }
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="DnnTransformer"]/*' />
    public sealed class DnnEstimator : IEstimator<DnnTransformer>
    {
        public enum Architecture
        {
            ResnetV2101,
            InceptionV3
        };

        public enum DnnFramework
        {
            Tensorflow
        };

        /// <summary>
        /// The options for the <see cref="DnnTransformer"/>.
        /// </summary>
        internal sealed class Options : TransformInputBase
        {
            /// <summary>
            /// Location of the TensorFlow model.
            /// </summary>
            [Argument(ArgumentType.Required, HelpText = "TensorFlow model used by the transform. Please see https://www.tensorflow.org/mobile/prepare_models for more details.", SortOrder = 0)]
            public string ModelLocation;

            /// <summary>
            /// The names of the model inputs.
            /// </summary>
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The names of the model inputs", ShortName = "inputs", SortOrder = 1)]
            public string[] InputColumns;

            /// <summary>
            /// The names of the requested model outputs.
            /// </summary>
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "The name of the outputs", ShortName = "outputs", SortOrder = 2)]
            public string[] OutputColumns;

            /// <summary>
            /// The name of the label column in <see cref="IDataView"/> that will be mapped to label node in TensorFlow model.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Training labels.", ShortName = "label", SortOrder = 4)]
            public string LabelColumn;

            /// <summary>
            /// The name of the label in TensorFlow model.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "TensorFlow label node.", ShortName = "TFLabel", SortOrder = 5)]
            public string TensorFlowLabel;

            /// <summary>
            /// Name of the operation in TensorFlow graph that is used for optimizing parameters in the graph.
            /// Usually it is the name specified in the minimize method of optimizer in python
            /// e.g. optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name = "SGDOptimizer").
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the optimization operation in the TensorFlow graph.", ShortName = "OptimizationOp", SortOrder = 6)]
            public string OptimizationOperation;

            /// <summary>
            /// The name of the operation in the TensorFlow graph to compute training loss (Optional).
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the operation in the TensorFlow graph to compute training loss (Optional)", ShortName = "LossOp", SortOrder = 7)]
            public string LossOperation;

            /// <summary>
            /// The name of the operation in the TensorFlow graph to compute performance metric during training (Optional).
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the operation in the TensorFlow graph to compute performance metric during training (Optional)", ShortName = "MetricOp", SortOrder = 8)]
            public string MetricOperation;

            /// <summary>
            /// Number of samples to use for mini-batch training.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of samples to use for mini-batch training.", SortOrder = 9)]
            public int BatchSize = 64;

            /// <summary>
            /// Number of training iterations.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of training iterations.", SortOrder = 10)]
            public int Epoch = 5;

            /// <summary>
            /// The name of the operation in the TensorFlow graph which sets optimizer learning rate (Optional).
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the operation in the TensorFlow graph which sets optimizer learning rate (Optional).", SortOrder = 11)]
            public string LearningRateOperation;

            /// <summary>
            /// Learning rate to use during optimization.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Learning rate to use during optimization.", SortOrder = 12)]
            public float LearningRate = 0.01f;

            /// <summary>
            /// Name of the input in TensorFlow graph that specifiy the location for saving/restoring models to/from disk.
            /// This parameter is set by different kinds of 'Savers' in TensorFlow and users don't have control over this.
            /// Therefore, its highly unlikely that this parameter is changed from its default value of 'save/Const'.
            /// Please change it cautiously if you need to.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input in TensorFlow graph that specifiy the location for saving/restoring models from disk.", SortOrder = 13)]
            public string SaveLocationOperation = "save/Const";

            /// <summary>
            /// Name of the operation in TensorFlow graph that is used for saving/restoring models to/from disk.
            /// This parameter is set by different kinds of 'Savers' in TensorFlow and users don't have control over this.
            /// Therefore, its highly unlikely that this parameter is changed from its default value of 'save/control_dependency'.
            /// Please change it cautiously if you need to.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input in TensorFlow graph that specifiy the location for saving/restoring models from disk.", SortOrder = 14)]
            public string SaveOperation = "save/control_dependency";

            /// <summary>
            /// Needed for command line to specify if retraining is requested.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Retrain TensorFlow model.", SortOrder = 15)]
            public bool ReTrain = false;

            /// <summary>
            /// Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].
            /// </summary>
            /// <remarks>
            /// This parameter is used to deal with models that have unknown shape but the internal operators in the model require data to have batch dimension as well.
            /// In this case, there is no way to induce shape from the model's inputs or input data.
            /// </remarks>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Add a batch dimension to the input e.g. input = [224, 224, 3] => [-1, 224, 224, 3].", SortOrder = 16)]
            public bool AddBatchDimensionInputs = false;

            /// <summary>
            /// Indicates if transfer learning is requested.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Transfer learning on a model.", SortOrder = 15)]
            public bool TransferLearning = false;

            /// <summary>
            /// Specifies the model architecture to be used in the case of image classification training using transfer learning.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Model architecture to be used in transfer learning for image classification.", SortOrder = 15)]
            public Architecture Arch = Architecture.ResnetV2101;

            /// <summary>
            /// Name of the tensor that will contain the output scores of the last layer when transfer learning is done.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Softmax tensor of the last layer in transfer learning.", SortOrder = 15)]
            public string ScoreColumnName = "Scores";

            /// <summary>
            /// Name of the tensor that will contain the predicted label from output scores of the last layer when transfer learning is done.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Argmax tensor of the last layer in transfer learning.", SortOrder = 15)]
            public string PredictedLabelColumnName = "PredictedLabel";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Checkpoint folder to store graph files in the event of transfer learning.", SortOrder = 15)]
            public string CheckpointName = "_retrain_checkpoint";
        }

        private readonly IHost _host;
        private readonly Options _options;
        private readonly DnnModel _tensorFlowModel;
        private readonly TF_DataType[] _tfInputTypes;
        private readonly DataViewType[] _outputTypes;
        private DnnTransformer _transformer;

        internal DnnEstimator(IHostEnvironment env, Options options, DnnModel tensorFlowModel)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(DnnEstimator));
            _options = options;
            _tensorFlowModel = tensorFlowModel;

            if (options.TransferLearning)
                _tfInputTypes = new[] { TF_DataType.TF_FLOAT };
            else
            {
                var inputTuple = DnnTransformer.GetInputInfo(_host, tensorFlowModel.Session, options.InputColumns);
                _tfInputTypes = inputTuple.tfInputTypes;
            }
            if (options.TransferLearning)
                _outputTypes = new[] { new VectorDataViewType(NumberDataViewType.Single), new VectorDataViewType(NumberDataViewType.Single, 1) };
            else
                _outputTypes = DnnTransformer.GetOutputInfo(_host, tensorFlowModel.Session, options.OutputColumns).outputTypes;
        }

        private static Options CreateArguments(DnnModel tensorFlowModel, string[] outputColumnNames, string[] inputColumnName, bool addBatchDimensionInput)
        {
            var options = new Options();
            options.ModelLocation = tensorFlowModel.ModelPath;
            options.InputColumns = inputColumnName;
            options.OutputColumns = outputColumnNames;
            options.ReTrain = false;
            options.AddBatchDimensionInputs = addBatchDimensionInput;
            return options;
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            var resultDic = inputSchema.ToDictionary(x => x.Name);
            for (var i = 0; i < _options.InputColumns.Length; i++)
            {
                var input = _options.InputColumns[i];
                if (!inputSchema.TryFindColumn(input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
                if (!(col.Kind == SchemaShape.Column.VectorKind.Vector))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "vector", col.GetTypeString());
                var expectedType = DnnUtils.Tf2MlNetType(_tfInputTypes[i]);
                if (col.ItemType != expectedType)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, expectedType.ToString(), col.ItemType.ToString());
            }
            for (var i = 0; i < _options.OutputColumns.Length; i++)
            {
                resultDic[_options.OutputColumns[i]] = new SchemaShape.Column(_options.OutputColumns[i],
                    _outputTypes[i].IsKnownSizeVector() ? SchemaShape.Column.VectorKind.Vector
                    : SchemaShape.Column.VectorKind.VariableVector, _outputTypes[i].GetItemType(), false);
            }
            return new SchemaShape(resultDic.Values);
        }

        /// <summary>
        /// Trains and returns a <see cref="DnnTransformer"/>.
        /// </summary>
        public DnnTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            if (_transformer == null)
            {
                _transformer = _options.ReTrain || _options.TransferLearning ? new DnnTransformer(_host, _options, _tensorFlowModel, input) :
                    new DnnTransformer(_host, _tensorFlowModel.Session, _options.OutputColumns, _options.InputColumns,
                    DnnUtils.IsSavedModel(_host, _options.ModelLocation) ? _options.ModelLocation : null, false, _options.AddBatchDimensionInputs);
            }

            // Validate input schema.
            _transformer.GetOutputSchema(input.Schema);
            return _transformer;
        }

        /// <summary>
        /// Trains and returns a <see cref="DnnTransformer"/>.
        /// </summary>
        public DnnTransformer Fit(IDataView input, IDataView validationSet)
        {
            _host.CheckValue(input, nameof(input));
            if (_transformer == null)
            {
                _transformer = _options.ReTrain || _options.TransferLearning ? new DnnTransformer(_host, _options, _tensorFlowModel, input, validationSet) :
                    new DnnTransformer(_host, _tensorFlowModel.Session, _options.OutputColumns, _options.InputColumns,
                    DnnUtils.IsSavedModel(_host, _options.ModelLocation) ? _options.ModelLocation : null, false, _options.AddBatchDimensionInputs);
            }

            // Validate input schema.
            _transformer.GetOutputSchema(input.Schema);
            return _transformer;
        }
    }
}