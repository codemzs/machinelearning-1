// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using NumSharp;
using Tensorflow;

namespace Microsoft.ML.Transforms.TensorFlow
{
    internal static class TensorFlowUtils
    {
        /// <summary>
        /// Key to access operator's type (a string) in <see cref="DataViewSchema.Column.Annotations"/>.
        /// Its value describes the Tensorflow operator that produces this <see cref="DataViewSchema.Column"/>.
        /// </summary>
        internal const string TensorflowOperatorTypeKind = "TensorflowOperatorType";
        /// <summary>
        /// Key to access upstream operators' names (a string array) in <see cref="DataViewSchema.Column.Annotations"/>.
        /// Its value states operators that the associated <see cref="DataViewSchema.Column"/>'s generator depends on.
        /// </summary>
        internal const string TensorflowUpstreamOperatorsKind = "TensorflowUpstreamOperators";

        /*internal static DataViewSchema GetModelSchema(IExceptionContext ectx, Graph graph, string opType = null)
        {
            var schemaBuilder = new DataViewSchema.Builder();
            foreach (var op in c_api_util.tf_operations(graph))
            {
                if (opType != null && opType != op.OpType)
                    continue;

                var tfType = op.OutputType(0);
                // Determine element type in Tensorflow tensor. For example, a vector of floats may get NumberType.R4 here.
                var mlType = Tf2MlNetTypeOrNull(tfType);

                // If the type is not supported in ML.NET then we cannot represent it as a column in an Schema.
                // We also cannot output it with a TensorFlowTransform, so we skip it.
                // Furthermore, operators which have NumOutputs <= 0 needs to be filtered.
                // The 'GetTensorShape' method crashes TensorFlow runtime
                // (https://github.com/dotnet/machinelearning/issues/2156) when the operator has no outputs.
                if (mlType == null || op.NumOutputs <= 0)
                    continue;

                // Construct the final ML.NET type of a Tensorflow variable.
                var tensorShape = graph. .GetTensorShape(op[0]).ToIntArray();
                var columnType = new VectorDataViewType(mlType);
                if (!(Utils.Size(tensorShape) == 1 && tensorShape[0] <= 0) &&
                    (Utils.Size(tensorShape) > 0 && tensorShape.Skip(1).All(x => x > 0)))
                    columnType = new VectorDataViewType(mlType, tensorShape[0] > 0 ? tensorShape : tensorShape.Skip(1).ToArray());

                // There can be at most two metadata fields.
                //  1. The first field always presents. Its value is this operator's type. For example,
                //     if an output is produced by an "Softmax" operator, the value of this field should be "Softmax".
                //  2. The second field stores operators whose outputs are consumed by this operator. In other words,
                //     these values are names of some upstream operators which should be evaluated before executing
                //     the current operator. It's possible that one operator doesn't need any input, so this field
                //     can be missing.
                var metadataBuilder = new DataViewSchema.Annotations.Builder();
                // Create the first metadata field.
                metadataBuilder.Add(TensorflowOperatorTypeKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => value = op.OpType.AsMemory());
                if (op.NumInputs > 0)
                {
                    // Put upstream operators' names to an array (type: VBuffer) of string (type: ReadOnlyMemory<char>).
                    VBuffer<ReadOnlyMemory<char>> upstreamOperatorNames = default;
                    var bufferEditor = VBufferEditor.Create(ref upstreamOperatorNames, op.NumInputs);
                    for (int i = 0; i < op.NumInputs; ++i)
                        bufferEditor.Values[i] = c_api.TF_OperationName(op.GetInput(i).oper).GetStr().AsMemory();
                    upstreamOperatorNames = bufferEditor.Commit(); // Used in metadata's getter.

                    // Create the second metadata field.
                    metadataBuilder.Add(TensorflowUpstreamOperatorsKind, new VectorDataViewType(TextDataViewType.Instance, op.NumInputs),
                        (ref VBuffer<ReadOnlyMemory<char>> value) => { upstreamOperatorNames.CopyTo(ref value); });
                }

                schemaBuilder.AddColumn(op.Name, columnType, metadataBuilder.ToAnnotations());
            }
            return schemaBuilder.ToSchema();
        }

        /// <summary>
        /// This method retrieves the information about the graph nodes of a TensorFlow model as an <see cref="DataViewSchema"/>.
        /// For every node in the graph that has an output type that is compatible with the types supported by
        /// <see cref="TfTransferLearningTransformer"/>, the output schema contains a column with the name of that node, and the
        /// type of its output (including the item type and the shape, if it is known). Every column also contains metadata
        /// of kind <see cref="TensorflowOperatorTypeKind"/>, indicating the operation type of the node, and if that node has inputs in the graph,
        /// it contains metadata of kind <see cref="TensorflowUpstreamOperatorsKind"/>, indicating the names of the input nodes.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">Model to load.</param>
        internal static DataViewSchema GetModelSchema(IHostEnvironment env, string modelPath)
        {
            var model = LoadTensorFlowModel(env, modelPath);
            return GetModelSchema(env, model.Session.graph);
        }*/

        internal static PrimitiveDataViewType Tf2MlNetType(TF_DataType type)
        {
            var mlNetType = Tf2MlNetTypeOrNull(type);
            if (mlNetType == null)
                throw new NotSupportedException("TensorFlow type not supported.");
            return mlNetType;
        }

        private static PrimitiveDataViewType Tf2MlNetTypeOrNull(TF_DataType type)
        {
            switch (type)
            {
                case TF_DataType.TF_FLOAT:
                    return NumberDataViewType.Single;
                case TF_DataType.DtFloatRef:
                    return NumberDataViewType.Single;
                case TF_DataType.TF_DOUBLE:
                    return NumberDataViewType.Double;
                case TF_DataType.TF_UINT8:
                    return NumberDataViewType.Byte;
                case TF_DataType.TF_UINT16:
                    return NumberDataViewType.UInt16;
                case TF_DataType.TF_UINT32:
                    return NumberDataViewType.UInt32;
                case TF_DataType.TF_UINT64:
                    return NumberDataViewType.UInt64;
                case TF_DataType.TF_INT8:
                    return NumberDataViewType.SByte;
                case TF_DataType.TF_INT16:
                    return NumberDataViewType.Int16;
                case TF_DataType.TF_INT32:
                    return NumberDataViewType.Int32;
                case TF_DataType.TF_INT64:
                    return NumberDataViewType.Int64;
                case TF_DataType.TF_BOOL:
                    return BooleanDataViewType.Instance;
                case TF_DataType.TF_STRING:
                    return TextDataViewType.Instance;
                default:
                    return null;
            }
        }

        internal static Session LoadTFSession(IExceptionContext ectx, byte[] modelBytes, string modelFile = null)
        {
            var graph = new Graph();
            try
            {
                graph.Import(modelBytes);
            }
            catch (Exception ex)
            {
                if (!string.IsNullOrEmpty(modelFile))
                    throw ectx.Except($"TensorFlow exception triggered while loading model from '{modelFile}'");
#pragma warning disable MSML_NoMessagesForLoadContext
                throw ectx.ExceptDecode(ex, "Tensorflow exception triggered while loading model.");
#pragma warning restore MSML_NoMessagesForLoadContext

            }
            return new Session(graph);
        }

        private static Session LoadTFSession(IHostEnvironment env, string exportDirSavedModel)
        {
            Contracts.Check(env != null, nameof(env));
            env.CheckValue(exportDirSavedModel, nameof(exportDirSavedModel));

            return Session.LoadFromSavedModel(exportDirSavedModel);
        }

        // A TensorFlow frozen model is a single file. An un-frozen (SavedModel) on the other hand has a well-defined folder structure.
        // Given a modelPath, this utility method determines if we should treat it as a SavedModel or not
        internal static bool IsSavedModel(IHostEnvironment env, string modelPath)
        {
            Contracts.Check(env != null, nameof(env));
            env.CheckNonWhiteSpace(modelPath, nameof(modelPath));
            FileAttributes attr = File.GetAttributes(modelPath);
            return attr.HasFlag(FileAttributes.Directory);
        }

        // Currently used in TensorFlowTransform to protect temporary folders used when working with TensorFlow's SavedModel format.
        // Models are considered executable code, so we need to ACL tthe temp folders for high-rights process (so low-rights process can’t access it).
        /// <summary>
        ///  Given a folder path, create it with proper ACL if it doesn't exist.
        ///  Fails if the folder name is empty, or can't create the folder.
        /// </summary>
        internal static void CreateFolderWithAclIfNotExists(IHostEnvironment env, string folder)
        {
            Contracts.Check(env != null, nameof(env));
            env.CheckNonWhiteSpace(folder, nameof(folder));

            //if directory exists, do nothing.
            if (Directory.Exists(folder))
                return;

            WindowsIdentity currentIdentity = null;
            try
            {
                currentIdentity = WindowsIdentity.GetCurrent();
            }
            catch (PlatformNotSupportedException)
            { }

            if (currentIdentity != null && new WindowsPrincipal(currentIdentity).IsInRole(WindowsBuiltInRole.Administrator))
            {
                // Create high integrity dir and set no delete policy for all files under the directory.
                // In case of failure, throw exception.
                CreateTempDirectoryWithAcl(folder, currentIdentity.User.ToString());
            }
            else
            {
                try
                {
                    Directory.CreateDirectory(folder);
                }
                catch (Exception exc)
                {
                    throw Contracts.ExceptParam(nameof(folder), $"Failed to create folder for the provided path: {folder}. \nException: {exc.Message}");
                }
            }
        }

        internal static void DeleteFolderWithRetries(IHostEnvironment env, string folder)
        {
            Contracts.Check(env != null, nameof(env));
            int currentRetry = 0;
            int maxRetryCount = 10;
            using (var ch = env.Start("Delete folder"))
            {
                for (; ; )
                {
                    try
                    {
                        currentRetry++;
                        Directory.Delete(folder, true);
                        break;
                    }
                    catch (IOException e)
                    {
                        if (currentRetry > maxRetryCount)
                            throw;
                        ch.Info("Error deleting folder. {0}. Retry,", e.Message);
                    }
                }
            }
        }

        private static void CreateTempDirectoryWithAcl(string folder, string identity)
        {
            // Dacl Sddl string:
            // D: Dacl type
            // D; Deny access
            // OI; Object inherit ace
            // SD; Standard delete function
            // wIdentity.User Sid of the given user.
            // A; Allow access
            // OICI; Object inherit, container inherit
            // FA File access
            // BA Built-in administrators
            // S: Sacl type
            // ML;; Mandatory Label
            // NW;;; No write policy
            // HI High integrity processes only
            string sddl = "D:(D;OI;SD;;;" + identity + ")(A;OICI;FA;;;BA)S:(ML;OI;NW;;;HI)";

            try
            {
                var dir = Directory.CreateDirectory(folder);
                DirectorySecurity dirSec = new DirectorySecurity();
                dirSec.SetSecurityDescriptorSddlForm(sddl);
                dirSec.SetAccessRuleProtection(true, false);  // disable inheritance
                dir.SetAccessControl(dirSec);

                // Cleaning out the directory, in case someone managed to sneak in between creation and setting ACL.
                DirectoryInfo dirInfo = new DirectoryInfo(folder);
                foreach (FileInfo file in dirInfo.GetFiles())
                {
                    file.Delete();
                }
                foreach (DirectoryInfo subDirInfo in dirInfo.GetDirectories())
                {
                    subDirInfo.Delete(true);
                }
            }
            catch (Exception exc)
            {
                throw Contracts.ExceptParam(nameof(folder), $"Failed to create folder for the provided path: {folder}. \nException: {exc.Message}");
            }
        }

        /// <summary>
        /// Load TensorFlow model into memory.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="modelPath">The model to load.</param>
        /// <returns></returns>
        internal static TensorFlowModel LoadTensorFlowModel(IHostEnvironment env, string modelPath)
        {
            var session = GetSession(env, modelPath);
            return new TensorFlowModel(env, session, modelPath);
        }

        internal static Session GetSession(IHostEnvironment env, string modelPath)
        {
            Contracts.Check(env != null, nameof(env));
            if (IsSavedModel(env, modelPath))
            {
                env.CheckUserArg(Directory.Exists(modelPath), nameof(modelPath));
                return LoadTFSession(env, modelPath);
            }

            env.CheckUserArg(File.Exists(modelPath), nameof(modelPath));
            var bytes = File.ReadAllBytes(modelPath);
            return LoadTFSession(env, bytes, modelPath);
        }

        internal static unsafe void FetchData<T>(T[] data, Span<T> result)
        {
            var dataSpan = new Span<T>(data, 0, result.Length);
            dataSpan.CopyTo(result);
        }

        internal static unsafe void FetchStringData<T>(Tensor tensor, Span<T> result)
        {
            if (tensor == null)
                throw Contracts.ExceptEmpty(nameof(tensor));
            //
            // TF_STRING tensors are encoded with a table of 8-byte offsets followed by TF_StringEncode-encoded bytes.
            // [offset1, offset2,...,offsetn, s1size, s1bytes, s2size, s2bytes,...,snsize,snbytes]
            //
            long size = 1;
            foreach (var s in tensor.TensorShape.Dimensions)
                size *= s;

            var buffer = new byte[size][];
            var src = c_api.TF_TensorData(tensor);
            var srcLen = (IntPtr)(src.ToInt64() + (long)tensor.bytesize);
            src += (int)(size * 8);
            for (int i = 0; i < buffer.Length; i++)
            {
                using (var status = new Status())
                {
                    IntPtr dst = IntPtr.Zero;
                    ulong dstLen = 0;
                    var read = c_api.TF_StringDecode(src, (ulong)(srcLen.ToInt64() - src.ToInt64()), dst, ref dstLen, status);
                    status.Check();
                    buffer[i] = new byte[(int)dstLen];
                    Marshal.Copy(dst, buffer[i], 0, buffer[i].Length);
                    src += (int)read;
                }
            }

            for (int i = 0; i < buffer.Length; i++)
                result[i] = (T)(object)Encoding.UTF8.GetString(buffer[i]).AsMemory();
        }

        internal static bool IsTypeSupported(TF_DataType tfoutput)
        {
            switch (tfoutput)
            {
                case TF_DataType.TF_FLOAT:
                case TF_DataType.TF_DOUBLE:
                case TF_DataType.TF_UINT8:
                case TF_DataType.TF_UINT16:
                case TF_DataType.TF_UINT32:
                case TF_DataType.TF_UINT64:
                case TF_DataType.TF_INT8:
                case TF_DataType.TF_INT16:
                case TF_DataType.TF_INT32:
                case TF_DataType.TF_INT64:
                case TF_DataType.TF_BOOL:
                case TF_DataType.TF_STRING:
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        /// Use the runner class to easily configure inputs, outputs and targets to be passed to the session runner.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
        /// to construct the parameters that will be passed to the TFSession.Run method.
        /// </para>
        /// <para>
        /// Instances of this class are created by calling the GetRunner method on the TFSession.
        /// </para>
        /// <para>
        /// The various methods in this class return an instance to the Runner itsel, to allow
        /// to easily construct chains of execution like this:
        /// </para>
        /// <code>
        /// var result = session.GetRunner ().AddINput (myInput).Fetch (MyOutput).Run ();
        /// </code>
        /// <para>
        /// You do not need to chain the operations, this works just the same:
        /// </para>
        /// <code>
        /// runner = session.GetRunner ();
        /// runner.AddInput(myInput);
        /// runner.Fetch(myOutput);
        /// var results = runner.Run();
        /// </code>
        /// </remarks>
        public class Runner
        {
            private List<TF_Input> _inputs;
            private List<TF_Output> _outputs;
            private List<Tensor> _inputValues;
            private List<Operation> _targets;
            private Session _session;

            internal Runner(Session session)
            {
                _inputs = new List<TF_Input>();
                _outputs = new List<TF_Output>();
                _inputValues = new List<Tensor>();
                _targets = new List<Operation>();
                _session = session;
                RunMetadata = null;
                RunOptions = null;
            }

            /// <summary>
            /// Adds an input to the session
            /// </summary>
            /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
            /// <param name="input">Incoming port.</param>
            /// <param name="value">Value to assing to the incoming port.</param>
            public Runner AddInput(TF_Input input, Tensor value)
            {
                if (value == null)
                    throw new ArgumentNullException(nameof(value));
                _inputs.Add(input);
                _inputValues.Add(value);
                return this;
            }

            /// <summary>
            /// Adds an input to the session specified by name, with an optional index in the operation (separated by a colon).
            /// </summary>
            /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
            /// <param name="input">Incoming port, with an optional index separated by a colon.</param>
            /// <param name="value">Value to assing to the incoming port.</param>
            public Runner AddInput(string input, Tensor value)
            {
                if (value == null)
                    throw new ArgumentNullException(nameof(value));
                _inputs.Add(ParseInput(input));
                _inputValues.Add(value);
                return this;
            }

            /// <summary>
            /// Adds the specified operations as the ones to be retrieved.
            /// </summary>
            /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
            /// <param name="targets">One or more targets.</param>
            public Runner AddTarget(params Operation[] targets)
            {
                foreach (var t in targets)
                    _targets.Add(t);
                return this;
            }

            private TF_Input ParseInput(string operation)
            {
                var p = operation.IndexOf(':');
                if (p != -1 && p != operation.Length - 1)
                {
                    var op = operation.Substring(0, p);
                    if (int.TryParse(operation.Substring(p + 1), out var idx))
                    {
                        return new TF_Input(_session.graph.OperationByName(op), idx);
                    }
                }
                return new TF_Input(_session.graph.OperationByName(operation), 0);
            }

            // Parses user strings that contain both the operation name and an index.
            private TF_Output ParseOutput(string operation)
            {
                var p = operation.IndexOf(':');
                if (p != -1 && p != operation.Length - 1)
                {
                    var op = operation.Substring(0, p);
                    if (int.TryParse(operation.Substring(p + 1), out var idx))
                    {
                        return new TF_Output(_session.graph.OperationByName(op), idx);
                    }
                }
                return new TF_Output(_session.graph.OperationByName(operation), 0);
            }

            /// <summary>
            /// Adds the specified operation names as the ones to be retrieved.
            /// </summary>
            /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
            /// <param name="targetNames">One or more target names.</param>
            public Runner AddTarget(params string[] targetNames)
            {
                foreach (var tn in targetNames)
                    _targets.Add(_session.graph.OperationByName(tn));
                return this;
            }

            /// <summary>
            /// Makes the Run method return the index-th output of the tensor referenced by operation.
            /// </summary>
            /// <returns>The instance of runner, to allow chaining operations.</returns>
            /// <param name="operation">The name of the operation in the graph.</param>
            /// <param name="index">The index of the output in the operation.</param>
            public Runner Fetch(string operation, int index)
            {
                var op = _session.graph.OperationByName(operation);
                _outputs.Add(new TF_Output(op, index));
                return this;
            }

            /// <summary>
            /// Makes the Run method return the output of the tensor referenced by operation, the operation string can contain the output index.
            /// </summary>
            /// <returns>The instance of runner, to allow chaining operations.</returns>
            /// <param name="operation">The name of the operation in the graph, which might be a simple name, or it might be name:index,
            /// where the index is the .</param>
            public Runner Fetch(string operation)
            {
                var op = ParseOutput(operation);
                _outputs.Add(op);
                return this;
            }

            /// <summary>
            /// Makes the Run method return the output of the tensor referenced by output
            /// </summary>
            /// <returns>The instance of runner, to allow chaining operations.</returns>
            /// <param name="output">The output referencing a specified tensor.</param>
            public Runner Fetch(TF_Output output)
            {
                _outputs.Add(output);
                return this;
            }

            /// <summary>
            /// Makes the Run method return the output of all the tensor referenced by outputs.
            /// </summary>
            /// <returns>The instance of runner, to allow chaining operations.</returns>
            /// <param name="outputs">The outputs referencing a specified tensor.</param>
            public Runner Fetch(params TF_Output[] outputs)
            {
                foreach (var output in outputs)
                    _outputs.Add(output);
                return this;
            }

            /// <summary>
            /// Makes the Run method return the output of all the tensor referenced by outputs.
            /// </summary>
            /// <returns>The instance of runner, to allow chaining operations.</returns>
            /// <param name="outputs">The output sreferencing a specified tensor.</param>
            public Runner Fetch(params string[] outputs)
            {
                foreach (var output in outputs)
                    _outputs.Add(ParseOutput(output));
                return this;
            }

            /// <summary>
            /// Protocol buffer encoded block containing the metadata passed to the <see cref="M:TensorFlow.TFSession.Run"/> method.
            /// </summary>
            public Tensorflow.Buffer RunMetadata;

            /// <summary>
            /// Protocol buffer encoded block containing the run options passed to the <see cref="M:TensorFlow.TFSession.Run"/> method.
            /// </summary>
            public Tensorflow.Buffer RunOptions;

            /// <summary>
            ///  Execute the graph fragments necessary to compute all requested fetches.
            /// </summary>
            /// <returns>One TFTensor for each call to Fetch that you made, in the order that you made them.</returns>
            /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
            public Tensor[] Run(Status status = null)
            {
                FeedItem[] items = new FeedItem[_inputValues.Count];
                for(int index = 0; index < _inputValues.Count; index++)
                {
                    items[index] = new FeedItem(_inputValues[index], new NDArray(_inputValues[index].Data(), _inputValues[index].TensorShape));
                }

                //return _session.run(_inputs.ToArray(), _inputValues.ToArray(), _outputs.ToArray(), _targets.ToArray(), RunMetadata, RunOptions, status);
                return null;
            }

            /// <summary>
            /// Run the specified operation, by adding it implicity to the output, single return value
            /// </summary>
            /// <param name="operation">The output of the operation.</param>
            /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
            /// <remarks>
            /// This method is a convenience method, and when you call it, it will clear any
            /// calls that you might have done to Fetch() and use the specified operation to Fetch
            /// instead.
            /// </remarks>
            public Tensor Run(TF_Output operation, Status status = null)
            {
                _outputs.Clear();
                Fetch(operation);
                return Run(status)[0];
            }

        }

    }
}
