// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Dnn;

namespace Microsoft.ML
{
    /// <include file='doc.xml' path='doc/members/member[@name="DnnTransformer"]/*' />
    public static class DnnCatalog
    {
        /// <summary>
        /// Load TensorFlow model into memory. This is the convenience method that allows the model to be loaded once and subsequently use it for querying schema and creation of
        /// <see cref="DnnEstimator"/> using <see cref="DnnModel.ScoreTensorFlowModel(string, string, bool)"/>.
        /// </summary>
        /// <param name="catalog">The transform's catalog.</param>
        /// <param name="modelLocation">Location of the TensorFlow model.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[LoadTensorFlowModel](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/TensorFlow/TextClassification.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static DnnModel LoadDnnModel(this ModelOperationsCatalog catalog, string modelLocation, bool metaGraph = false)
            => DnnUtils.LoadDnnModel(CatalogUtils.GetEnvironment(catalog), modelLocation, metaGraph);
    }
}
