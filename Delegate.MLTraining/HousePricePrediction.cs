using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace Delegate.MLTraining
{
    class HousePricePrediction
    {
        [ColumnName("Score")]
        public float SalePrice;
    }
}
