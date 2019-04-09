using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace Samples.Dynamic
{
    public static class DetectIidSpike
    {
        class IidSpikeData
        {
            public float Value;

            public IidSpikeData(float value)
            {
                Value = value;
            }
        }

        class IidSpikePrediction
        {
            [VectorType(3)]
            public double[] Prediction { get; set; }
        }

        // This example creates a time series (list of Data with the i-th element corresponding to the i-th time slot). 
        // The estimator is applied then to identify spiking points in the series.
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Generate sample series data with a spike
            const int Size = 10;
            var data = new List<IidSpikeData>(Size);
            for (int i = 0; i < Size / 2; i++)
                data.Add(new IidSpikeData(5));
            // This is a spike
            data.Add(new IidSpikeData(10));
            for (int i = 0; i < Size / 2; i++)
                data.Add(new IidSpikeData(5));

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup IidSpikeDetector arguments
            string outputColumnName = nameof(IidSpikePrediction.Prediction);
            string inputColumnName = nameof(IidSpikeData.Value);
            // The transformed model.
            ITransformer model = ml.Transforms.DetectIidSpike(outputColumnName, inputColumnName, 95, Size).Fit(dataView);

            // Create a time series prediction engine from the model.
            var engine = model.CreateTimeSeriesPredictionFunction<IidSpikeData, IidSpikePrediction>(ml);

            // Create non-anomalous data and check for anomaly.
            for (int index = 0; index < 5; index++)
            {
                // Anomaly spike detection.
                var prediction = engine.Predict(new IidSpikeData(5));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", 5, prediction.Prediction[0],
                    prediction.Prediction[1], prediction.Prediction[2]);
            }

            // Spike.
            var spikePrediction = engine.Predict(new IidSpikeData(10));
            Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", 10, spikePrediction.Prediction[0],
                spikePrediction.Prediction[1], spikePrediction.Prediction[2]);

            // Checkpoint the model.
            var modelPath = "temp.zip";
            engine.CheckPoint(ml, modelPath);

            // Load the model.
            using (var file = File.OpenRead(modelPath))
                model = ml.Model.Load(file, out DataViewSchema schema);

            for (int index = 0; index < 5; index++)
            {
                // Anomaly spike detection.
                var prediction = engine.Predict(new IidSpikeData(5));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}", 5, prediction.Prediction[0],
                    prediction.Prediction[1], prediction.Prediction[2]);
            }

            // Data Alert   Score   P-Value
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 10     1      10.00    0.00  <-- alert is on, predicted spike (check-point model)
            // 5      0       5.00    0.26  <-- load model from disk.
            // 5      0       5.00    0.26
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
            // 5      0       5.00    0.50
        }
    }
}
