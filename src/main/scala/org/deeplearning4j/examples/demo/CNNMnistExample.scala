package org.deeplearning4j.examples.demo

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4s.Implicits._
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

object CNNMnistExample {

  lazy val log = LoggerFactory.getLogger(CNNMnistExample.getClass)
  lazy val numRows = 28
  lazy val numColumns = 28
  lazy val nChannels = 1
  lazy val seed = 123

  def datasetIterator(batchSize: Int, numSamples: Int) = new MnistDataSetIterator(batchSize, numSamples, true)

  def buildNetwork(batchSize: Int, numSamples: Int, iterations: Int, outputNum: Int): MultiLayerNetwork = {
    log.info("Build model....")

    val builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .regularization(true).l2(0.0005)
      .learningRate(0.1)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.ADAGRAD)
      .list(6)
      .layer(0, new ConvolutionLayer.Builder(5, 5)
      .nIn(nChannels)
      .stride(1, 1)
      .nOut(20)
      .weightInit(WeightInit.XAVIER)
      .activation("relu")
      .build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
      .build())
      .layer(2, new ConvolutionLayer.Builder(5, 5)
      .nIn(20)
      .nOut(50)
      .stride(2, 2)
      .weightInit(WeightInit.XAVIER)
      .activation("relu")
      .build())
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
      .build())
      .layer(4, new DenseLayer.Builder().activation("relu")
      .weightInit(WeightInit.XAVIER)
      .nOut(200).build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
      .nOut(outputNum)
      .weightInit(WeightInit.XAVIER)
      .activation("softmax")
      .build())
      .backprop(true).pretrain(false)
    new ConvolutionLayerSetup(builder, 28, 28, 1)

    new MultiLayerNetwork(builder.build())
  }

  def main(args: Array[String]) = {
    val outputNum = 10
    val numSamples = 10000 //originally 60,000 but it takes a bit in CPU.
    val batchSize = 50
    val iterations = 1
    val listenerFreq = iterations / 5
    val splitTrainNum = (batchSize * .8).toInt

    val model = buildNetwork(batchSize, numSamples, iterations, outputNum)
    model.init()
    model.setListeners(new ScoreIterationListener(listenerFreq))

    log.info("Load data....")
    val mnistIter = datasetIterator(batchSize, numSamples)

    log.info("Train model....")
    //Do not load all the data into memory at once.
    mnistIter.asScala.foreach { mnist =>
      prettyPrint(mnist.getLabels,mnist.getFeatureMatrix)
      model.fit(mnist)
    }

    log.info("Evaluate model....")
    val testIter = datasetIterator(batchSize, 1000)
    val eval = new Evaluation(outputNum)
    testIter.asScala.foreach { mnist =>
      val output = model.output(mnist.getFeatureMatrix)
      eval.eval(mnist.getLabels, output)
    }

    log.info(eval.stats())
    log.info("****************Example finished********************")
  }

  def prettyPrint(label:INDArray, feature: INDArray): Unit = {
    val indArray = feature.getRow(0).reshape(28, 28) //same as `feature(0, 0 ->).reshape(28, 28)` or `feature(0, -> ).reshape(28, 28)`
    val i = Nd4j.argMax(label.getRow(0),1).sumT.toInt
    //filter out zero-filled rows for ease
    val filtered = for {row <- indArray.rowP if row.sumT > 0.01} yield row
    println("***********************************************************************************************************************************************")
    println(filtered)
    println(s"is trained with $i label. **********************************************************************************************************************")
  }
}
