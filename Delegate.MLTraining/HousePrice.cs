using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace Delegate.MLTraining
{
    class HousePrice
    {
        [Column("0")]
        public string Id;
        [Column("1")]
        public string MSSubClass;
        //[Column("3")]
        //public float LotFrontage;
        [Column("4")]
        public float LotArea;

        [Column("20")]
        public float YearRemodAdd;

        [Column("77")]
        public float YrSold;

        [Column("46")]
        public float GrLivArea;

        [Column("80", name: "Label")]        
        public float SalePrice;


    }
}
