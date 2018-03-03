mport org.apache.log4j.{Level,Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString,Word2Vec,StringIndexer}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext,SparkConf}

object SpamMessageClassifier{
    def main(args: Array[String]){
        Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
        Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
        
        if(args.length != 1){
            println("Usage: /path/to/spark/bin/spark-submit --master spark://master:9000" +
                "--driver-memory 1g --class chapter9.SpamMessageClassifier" +
                "sparklearning.jar MessageDataPath")
            sys.exit(1)
        }
        
        val conf = new SparkConf().setAppName("SpamMessageClassifier").setMaster("local[4]")
        val sc = new SparkContext(conf)
        val sqlCtx = new SQLContext(sc)
        
        val messageRDD = sc.textFile(args(0)).map(_.split("\t")).map(line => {(line(0),line(1).split(" "))})
        val smsDF = sqlCtx.createDataFrame(messageRDD).toDF("labelCol","contextCol")
        
        val labelIndexer = new SringIndexer()
            .setInputCol("labelCol")
            .setOutputCol("indexedLabelCol")
            .fit(smsDF)
            
        val word2Vec = new word2Vec()
            .setInputCol("contextCol")
            .setOutputCol("featuresCol")
            .setVectorSize(100)
            .setMinCount(1)
            
        val layers = Array[Int](100,6,5,2)
        
        val mpc = new MultilayerPerceptronClassifier()
            .setLayer(layers)
            .setBlockSize(512)
            .setSeed(1234L)
            .setMaxIter(128)
            .setFeaturesCol("featuresCol")
            .setLabelCol("indexedLabelCol")
            .setPredictionCol("predictionCol")
            
        val labelConverter = new IndexToString()
            .setInputCol("predictionCol")
            .setOutputCol("predictedLabelCol")
            .setLabels(labelIndexer.labels)
            
        val Array(trainingData, testData) = smsDF.randomSplit(Array(0.8,0.2))
        
        val pipeline = new Pipeline().setStages(Array(labelIndexer, word2Vec, mpc, labelConverter))
        val model = pipeline.fit(trainingData)
        val preResultDF = model.transform(testData)
        
        preResultDF.select("contextCol", "labelCol", "predictedLabelCol").show(20)
        
        val evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("indexedLabelCol")
            .setPredictionCol("predictionCol")
        val predictionAccuracy = evaluator.evaluate(preResultDF)
        println("Testing Accuracy is %2.4f".format(predictionAccuracy * 100) + "%")
        sc.stop
    }
}
