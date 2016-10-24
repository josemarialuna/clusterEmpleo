package es.us.empleo


import org.apache.spark.mllib.clustering.KMeansEmpleo
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}


/**
  *
  * @author José María Luna
  * @version 1.0
  * @since v1.0 Dev
  */
object MainKMeansEmpleo {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("Empleo Spark")
    val sc = new SparkContext(conf)

    //val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val fileOriginal = "C:\\datasets\\trabajadores.csv"
    val fileOriginalMin = "C:\\datasets\\trabajadores-min.csv"

    val fileKmeansDefault = "C:\\datasets\\kmeans_data.txt"
    //    val fileSource = args(0)
    var origen: String = fileOriginal
    var destino: String = Utils.whatTimeIsIt()
    var numClusters = 256
    var numIterations = 100

    if (args.size > 1) {
      origen = args(0)
      destino = args(1)
      numClusters = args(2).toInt
      numIterations = args(3).toInt
    }

    val data = sc.textFile(origen)
    //It skips the first line
    val skippedData = data.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }

    //val dataRDD = parsedData.map(s => Vectors.dense(s.split(';').map(_.toDouble))).cache()
    val dataRDD = skippedData.map(s => s.split(';').map(_.toDouble)).cache()

    // assuming dataRDD has type RDD[Array[Double]] and each Array has at least 4 items:
    val result = dataRDD
      .keyBy(_ (0).toInt)
      .mapValues(arr => Map(arr(1).toInt -> arr(2) / arr(3) * 100))
      .reduceByKey((a, b) => a ++ b)

    //result.saveAsObjectFile("Objeto")

    //val contAccu = sc.accumulator(0)
    val start = System.nanoTime()

    val porfin = result.mapValues { y =>
      val rowArray = new Array[Double](9281)
      for (z <- 1 to 9280) {
        if (y.exists(_._1 == z)) {
          rowArray(z) = y(z)
        } else {
          rowArray(z) = 0
        }
      }
      rowArray
    }

    //porfin.map(x=>(x._1,x._2.mkString(","))).saveAsTextFile(Utils.whatTimeIsIt())
    //porfin.saveAsObjectFile("fullMatrix")


    val parsedData = porfin.map(s => Vectors.dense(s._2)).cache()

    // Cluster the data into two classes using KMeans

    val clusters = KMeansEmpleo.train(parsedData, numClusters, numIterations)

    var elapsed = (System.nanoTime() - start) / 1000000000.0
    println("TIEMPO Clustering: " + elapsed)
    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // Save and load model
    //clusters.save(sc, "myModelSimil" + destino)
    //val sameModel = KMeansModel.load(sc, "myModelPath")

    val resultadoClustering = porfin.mapValues(arr => clusters.predict(Vectors.dense(arr))).cache()

    println("GUARDAMOS_trabajo")
    resultadoClustering.sortBy(_._1).saveAsTextFile(destino + "\\trabajo")

    val originalCluster = dataRDD
      .keyBy(_ (0).toInt)
      .mapValues(arr => arr(5).toInt)
      .distinct()
      .cache()

    println("GUARDAMOS_cardinalResultado")
    val rddResultadoCardinal = resultadoClustering.map(x => (x._2, 1)).reduceByKey(_ + _).sortBy(_._1).saveAsTextFile(destino + "\\cardinalResultado")
    val rddOriginalCardinal = originalCluster.map(x => (x._2, 1)).reduceByKey(_ + _).sortBy(_._1).saveAsTextFile(destino + "\\cardinalOriginal")

    val unitedCluster = originalCluster.join(resultadoClustering).map(_._2)
    val rddAccuracy = unitedCluster.map { case (x, y) => ((x, y), 1) }.reduceByKey((x, y) => x + y).sortBy(_._2)

    rddAccuracy.saveAsTextFile(destino + "\\comparativa")

    elapsed = (System.nanoTime() - start) / 1000000000.0
    println("TIEMPO TOTAL: " + elapsed)

    sc.stop()

  }

  //Return 0 if the data is empty, else return data parsed to Double
  def dataToDouble(s: String): Double = {
    return if (s.isEmpty) 0 else s.toDouble
  }

}

