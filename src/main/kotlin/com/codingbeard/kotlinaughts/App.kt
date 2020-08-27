package com.codingbeard.kotlinaughts

import
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.learning.config.Adam
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.optimize.listeners.ScoreIterationListener

/**
 * Hello world!
 */
object App {
    fun computeNotSuspend(model: MultiLayerNetwork): Unit {

        val batchSize = 128
        val rngSeed = (0..100).random()
        val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)
        val eval: org.nd4j.evaluation.classification.Evaluation = model.evaluate(mnistTest)
        println(eval.stats())
        return
    }

    @JvmStatic
    fun main(args: Array<String>) {
        val numRows = 28
        val numColumns = 28
        val pixelCount = numRows * numColumns
        val outputNum = 10
        val rngSeed = (0..100).random()
        val numEpochs = 5
        val batchSize = 128


        val multiLayerConfiguration = NeuralNetConfiguration.Builder()
                .seed(rngSeed.toLong())
                .updater(Adam())
                .list()
                .layer(
                        DenseLayer.Builder()
                                .nIn(pixelCount)
                                .nOut(1000)
                                .activation(Activation.RELU)
                                .build()
                )
                .layer(
                        OutputLayer.Builder() //create hidden layer
                                .nOut(outputNum)
                                .activation(Activation.SOFTMAX)
                                .build()
                )
                .build()

        val n = 10
        val model = MultiLayerNetwork(multiLayerConfiguration)
        val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
        for (i in 0 until numEpochs) {
            model.fit(mnistTrain)
        }
        model.init()
        val tasks = List(n) {model}

        tasks.stream().parallel().forEach { computeNotSuspend(it) }
    }
}