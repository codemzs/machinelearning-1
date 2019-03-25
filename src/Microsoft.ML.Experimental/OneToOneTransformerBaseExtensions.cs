﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Experimental
{
    public static class OneToOneTransformerBaseExtensions
    {
        /// <summary>
        /// Returns Input/Output column pair(s) for a <see cref="OneToOneTransformerBase"/>
        /// </summary>
        public static InputOutputColumnPair[] GetColumnPairs(this OneToOneTransformerBase transformer) => InputOutputColumnPair.ConvertFromValueTuples(transformer.ColumnPairs);
    }
}
