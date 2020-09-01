package com.codingbeard.kotlinaughts

import javafx.application.Application.launch
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.exp
import kotlin.random.Random

/**
 * Hello world!
 */
object App {
    fun getMoves(board: Array<Float>): List<Int> {
        var moves = mutableListOf<Int>()
        for (i in 0 until 9) {
            if (board[i] == 0f) {
                moves.add(i)
            }
        }

        return moves
    }

    fun getBoardState(board: Array<Float>): Float {
        //horizontal
        if (board[0] == 1f && board[1] == 1f && board[2] == 1f)
            return 1f
        if (board[3] == 1f && board[4] == 1f && board[5] == 1f)
            return 1f
        if (board[6] == 1f && board[7] == 1f && board[8] == 1f)
            return 1f
        if (board[0] == 2f && board[1] == 2f && board[2] == 2f)
            return 2f
        if (board[3] == 2f && board[4] == 2f && board[5] == 2f)
            return 2f
        if (board[6] == 2f && board[7] == 2f && board[8] == 2f)
            return 2f
        //vertical
        if (board[0] == 1f && board[3] == 1f && board[6] == 1f)
            return 1f
        if (board[1] == 1f && board[4] == 1f && board[7] == 1f)
            return 1f
        if (board[2] == 1f && board[5] == 1f && board[8] == 1f)
            return 1f
        if (board[0] == 2f && board[3] == 2f && board[6] == 2f)
            return 2f
        if (board[1] == 2f && board[4] == 2f && board[7] == 2f)
            return 2f
        if (board[2] == 2f && board[5] == 2f && board[8] == 2f)
            return 2f
        //diagonal
        if (board[0] == 1f && board[4] == 1f && board[8] == 1f)
            return 1f
        if (board[2] == 1f && board[4] == 1f && board[6] == 1f)
            return 1f
        if (board[0] == 2f && board[4] == 2f && board[8] == 2f)
            return 2f
        if (board[2] == 2f && board[4] == 2f && board[6] == 2f)
            return 2f
        return 0f
    }

    fun printBoard(board: Array<Float>) {
        var output = ""
        var row = ""
        for (i in 0..8) {
            if (i == 0) {
                row = ""
            } else if (i % 3 == 0) {
                output = "$output$row\n"
                row = ""
            }
            if (board[i] == 0f) {
                row = "$row -"
            } else if (board[i] == 1f) {
                row = "$row o"
            } else if (board[i] == 2f) {
                row = "$row x"
            }
        }

        output = "$output$row".trimIndent()
        println(output)
    }

    fun printRewards(board: Array<Float>) {
        var output = ""
        var row = ""
        for (i in 0..8) {
            if (i == 0) {
                row = ""
            } else if (i % 3 == 0) {
                output = "$output$row\n"
                row = ""
            }
            row = "$row " + board[i].toString()
        }

        output = "$output$row".trimIndent()
        println(output)
    }

    fun getMoveRewards(board: Array<Float>): List<Float> {
        var moves = mutableListOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        var naughts = 0
        var crosses = 0
        var turn = 1f
        var oppositeTurn = 2f

        var validMoveReward = 0.2f
        var invalidMoveReward = 0f
        var firstMoveReward = 0.8f
        var secondMoveReward = 0.8f
        var winningMoveReward = 0.7f
        var avoidingLossMoveReward = 0.8f
        var twoInARowReward = 0.1f

        for (i in 0..8) {
            if (board[i] == 0f) {
                moves[i] = validMoveReward
            } else {
                moves[i] = invalidMoveReward
            }

            if (board[i] == 1f) {
                naughts++
            }

            if (board[i] == 2f) {
                crosses++
            }
        }

        if (naughts == 0 && crosses == 0) {
            moves[4] = firstMoveReward
        }

        if ((naughts == 0 && crosses == 1) || (naughts == 1 && crosses == 0)) {
            if (board[4] == 0f) {
                moves[4] = secondMoveReward
            } else {
                moves[0] = secondMoveReward
                moves[2] = secondMoveReward
                moves[6] = secondMoveReward
                moves[8] = secondMoveReward
            }
        }

        if (naughts > crosses) {
            turn = 2f
            oppositeTurn = 1f
        }

        //horizontal 3rd free on turn
        if (board[0] == turn && board[1] == turn &&
                board[2] == 0f)
            moves[2] += winningMoveReward
        if (board[3] == turn && board[4] == turn &&
                board[5] == 0f)
            moves[5] += winningMoveReward
        if (board[6] == turn && board[7] == turn &&
                board[8] == 0f)
            moves[8] += winningMoveReward
        //horizontal 2nd free on turn
        if (board[0] == turn && board[1] == 0f &&
                board[2] == turn)
            moves[1] += winningMoveReward
        if (board[3] == turn && board[4] == 0f &&
                board[5] == turn)
            moves[4] += winningMoveReward
        if (board[6] == turn && board[7] == 0f &&
                board[8] == turn)
            moves[7] += winningMoveReward
        //horizontal 1st free on turn
        if (board[0] == 0f && board[1] == turn &&
                board[2] == turn)
            moves[0] += winningMoveReward
        if (board[3] == 0f && board[4] == turn &&
                board[5] == turn)
            moves[3] += winningMoveReward
        if (board[6] == 0f && board[7] == turn &&
                board[8] == turn)
            moves[6] += winningMoveReward
        //horizontal 3rd free off turn
        if (board[0] == oppositeTurn && board[1] == oppositeTurn &&
                board[2] == 0f)
            moves[2] += avoidingLossMoveReward
        if (board[3] == oppositeTurn && board[4] == oppositeTurn &&
                board[5] == 0f)
            moves[5] += avoidingLossMoveReward
        if (board[6] == oppositeTurn && board[7] == oppositeTurn &&
                board[8] == 0f)
            moves[8] += avoidingLossMoveReward
        //horizontal 2nd free off oppositeTurn
        if (board[0] == oppositeTurn && board[1] == 0f &&
                board[2] == oppositeTurn)
            moves[1] += avoidingLossMoveReward
        if (board[3] == oppositeTurn && board[4] == 0f &&
                board[5] == oppositeTurn)
            moves[4] += avoidingLossMoveReward
        if (board[6] == oppositeTurn && board[7] == 0f &&
                board[8] == oppositeTurn)
            moves[7] += avoidingLossMoveReward
        //horizontal 1st free off oppositeTurn
        if (board[0] == 0f && board[1] == oppositeTurn &&
                board[2] == oppositeTurn)
            moves[0] += avoidingLossMoveReward
        if (board[3] == 0f && board[4] == oppositeTurn &&
                board[5] == oppositeTurn)
            moves[3] += avoidingLossMoveReward
        if (board[6] == 0f && board[7] == oppositeTurn &&
                board[8] == oppositeTurn)
            moves[6] += avoidingLossMoveReward
        //vertical 3rd free on turn
        if (board[0] == turn && board[3] == turn &&
                board[6] == 0f)
            moves[6] += winningMoveReward
        if (board[1] == turn && board[4] == turn &&
                board[7] == 0f)
            moves[7] += winningMoveReward
        if (board[2] == turn && board[5] == turn &&
                board[8] == 0f)
            moves[8] += winningMoveReward
        //vertical 2nd free on turn
        if (board[0] == turn && board[3] == 0f &&
                board[6] == turn)
            moves[3] += winningMoveReward
        if (board[1] == turn && board[4] == 0f &&
                board[7] == turn)
            moves[4] += winningMoveReward
        if (board[2] == turn && board[5] == 0f &&
                board[8] == turn)
            moves[5] += winningMoveReward
        //vertical 1st free on turn
        if (board[0] == 0f && board[3] == turn &&
                board[6] == turn)
            moves[0] += winningMoveReward
        if (board[1] == 0f && board[4] == turn &&
                board[7] == turn)
            moves[1] += winningMoveReward
        if (board[2] == 0f && board[5] == turn &&
                board[8] == turn)
            moves[2] += winningMoveReward
        //vertical 3rd free off turn
        if (board[0] == oppositeTurn && board[3] == oppositeTurn &&
                board[6] == 0f)
            moves[6] += avoidingLossMoveReward
        if (board[1] == oppositeTurn && board[4] == oppositeTurn &&
                board[7] == 0f)
            moves[7] += avoidingLossMoveReward
        if (board[2] == oppositeTurn && board[5] == oppositeTurn &&
                board[8] == 0f)
            moves[8] += avoidingLossMoveReward
        //vertical 2nd free off oppositeTurn
        if (board[0] == oppositeTurn && board[3] == 0f &&
                board[6] == oppositeTurn)
            moves[3] += avoidingLossMoveReward
        if (board[1] == oppositeTurn && board[4] == 0f &&
                board[7] == oppositeTurn)
            moves[4] += avoidingLossMoveReward
        if (board[2] == oppositeTurn && board[5] == 0f &&
                board[8] == oppositeTurn)
            moves[5] += avoidingLossMoveReward
        //vertical 1st free off oppositeTurn
        if (board[0] == 0f && board[3] == oppositeTurn &&
                board[6] == oppositeTurn)
            moves[0] += avoidingLossMoveReward
        if (board[1] == 0f && board[4] == oppositeTurn &&
                board[7] == oppositeTurn)
            moves[1] += avoidingLossMoveReward
        if (board[2] == 0f && board[5] == oppositeTurn &&
                board[8] == oppositeTurn)
            moves[2] += avoidingLossMoveReward

        //diagonal 3rd free on turn
        if (board[0] == turn && board[4] == turn &&
                board[8] == 0f)
            moves[8] += winningMoveReward
        if (board[2] == turn && board[4] == turn &&
                board[6] == 0f)
            moves[6] += winningMoveReward
        //diagonal 2nd free on turn
        if (board[0] == turn && board[4] == 0f &&
                board[8] == turn)
            moves[4] += winningMoveReward
        if (board[2] == turn && board[4] == 0f &&
                board[6] == turn)
            moves[4] += winningMoveReward
        //diagonal 1st free on turn
        if (board[0] == 0f && board[4] == turn &&
                board[8] == turn)
            moves[0] += winningMoveReward
        if (board[2] == 0f && board[4] == turn &&
                board[6] == turn)
            moves[2] += winningMoveReward

        //diagonal 3rd free off turn
        if (board[0] == oppositeTurn && board[4] == oppositeTurn &&
                board[8] == 0f)
            moves[8] += avoidingLossMoveReward
        if (board[2] == oppositeTurn && board[4] == oppositeTurn &&
                board[6] == 0f)
            moves[6] += avoidingLossMoveReward
        //diagonal 2nd free off turn
        if (board[0] == oppositeTurn && board[4] == 0f &&
                board[8] == oppositeTurn)
            moves[4] += avoidingLossMoveReward
        if (board[2] == oppositeTurn && board[4] == 0f &&
                board[6] == oppositeTurn)
            moves[4] += avoidingLossMoveReward
        //diagonal 1st free off turn
        if (board[0] == 0f && board[4] == oppositeTurn &&
                board[8] == oppositeTurn)
            moves[0] += avoidingLossMoveReward
        if (board[2] == 0f && board[4] == oppositeTurn &&
                board[6] == oppositeTurn)
            moves[2] += avoidingLossMoveReward

        //horizontal 2nd, 3rd free on turn
        if (board[0] == turn && board[1] == 0f && board[2] == 0f) {
            moves[1] += twoInARowReward
            moves[2] += twoInARowReward
        }

        if (board[3] == turn && board[4] == 0f && board[5] == 0f) {
            moves[4] += twoInARowReward
            moves[5] += twoInARowReward
        }

        if (board[6] == turn && board[7] == 0f && board[8] == 0f) {
            moves[7] += twoInARowReward
            moves[8] += twoInARowReward
        }

        //horizontal 1st, 2nd free on turn
        if (board[0] == 0f && board[1] == 0f && board[2] == turn) {
            moves[0] += twoInARowReward
            moves[1] += twoInARowReward
        }

        if (board[3] == 0f && board[4] == 0f && board[5] == turn) {
            moves[3] += twoInARowReward
            moves[4] += twoInARowReward
        }

        if (board[6] == 0f && board[7] == 0f && board[8] == turn) {
            moves[6] += twoInARowReward
            moves[7] += twoInARowReward
        }

        //horizontal 1st, 3rd free on turn
        if (board[0] == 0f && board[1] == turn && board[2] == 0f) {
            moves[0] += twoInARowReward
            moves[2] += twoInARowReward
        }

        if (board[3] == 0f && board[4] == turn && board[5] == 0f) {
            moves[3] += twoInARowReward
            moves[5] += twoInARowReward
        }

        if (board[6] == 0f && board[7] == turn && board[8] == 0f) {
            moves[6] += twoInARowReward
            moves[8] += twoInARowReward
        }

        //vertical 2nd, 3rd free on turn
        if (board[0] == turn && board[3] == 0f && board[6] == 0f) {
            moves[3] += twoInARowReward
            moves[6] += twoInARowReward
        }

        if (board[1] == turn && board[4] == 0f && board[7] == 0f) {
            moves[4] += twoInARowReward
            moves[7] += twoInARowReward
        }

        if (board[2] == turn && board[5] == 0f && board[8] == 0f) {
            moves[5] += twoInARowReward
            moves[8] += twoInARowReward
        }

        //vertical 1st, 2nd free on turn
        if (board[0] == 0f && board[3] == 0f && board[6] == turn) {
            moves[0] += twoInARowReward
            moves[3] += twoInARowReward
        }

        if (board[1] == 0f && board[4] == 0f && board[7] == turn) {
            moves[1] += twoInARowReward
            moves[4] += twoInARowReward
        }

        if (board[2] == 0f && board[5] == 0f && board[8] == turn) {
            moves[2] += twoInARowReward
            moves[5] += twoInARowReward
        }

        //vertical 1st, 3rd free on turn
        if (board[0] == 0f && board[3] == turn && board[6] == 0f) {
            moves[0] += twoInARowReward
            moves[6] += twoInARowReward
        }

        if (board[1] == 0f && board[4] == turn && board[7] == 0f) {
            moves[1] += twoInARowReward
            moves[7] += twoInARowReward
        }

        if (board[2] == 0f && board[5] == turn && board[8] == 0f) {
            moves[2] += twoInARowReward
            moves[8] += twoInARowReward
        }

        //diagonal 2nd, 3rd free on turn
        if (board[0] == turn && board[4] == 0f && board[8] == 0f) {
            moves[4] += twoInARowReward
            moves[8] += twoInARowReward
        }

        if (board[2] == turn && board[4] == 0f && board[6] == 0f) {
            moves[4] += twoInARowReward
            moves[6] += twoInARowReward
        }

        //diagonal 1st, 2nd free on turn
        if (board[0] == 0f && board[4] == 0f && board[8] == turn) {
            moves[0] += twoInARowReward
            moves[4] += twoInARowReward
        }

        if (board[2] == 0f && board[4] == 0f && board[6] == turn) {
            moves[2] += twoInARowReward
            moves[4] += twoInARowReward
        }

        //diagonal 1st, 3rd free on turn
        if (board[0] == 0f && board[4] == turn && board[8] == 0f) {
            moves[0] += twoInARowReward
            moves[8] += twoInARowReward
        }

        if (board[2] == 0f && board[4] == turn && board[6] == 0f) {
            moves[2] += twoInARowReward
            moves[6] += twoInARowReward
        }

        var key = 0
        for (i in 0..8) {
            if (moves[i] > 1f) {
                moves[key] = 1f
            }
            key++
        }

        return moves
    }


    fun computeNotSuspend(i: Int, model: MultiLayerNetwork): Unit {
        val batchSize = 128
        val rngSeed = (0..100).random()
        val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)
        val eval: org.nd4j.evaluation.classification.Evaluation = model.evaluate(mnistTest)
        println(i)
        return
    }

    fun argMax(list: List<Float>): Int {
        var max = 0f
        var key = 0
        var maxKeys = mutableListOf<Int>()
        for (i in 0 until list.count()) {
            if (list[i] > max) {
                max = list[i]
            }
        }
        for (i in 0 until list.count()) {
            if (list[i] == max) {
                maxKeys.add(key)
            }

            key++
        }

        if (maxKeys.count() > 0) {
            return maxKeys.random()
        }

        return 0
    }

    fun argMax(list: INDArray): Int {
        var max = 0f
        var key = 0
        var maxKeys = mutableListOf<Int>()
        for (i in 0 until list.length()) {
            if (list.getFloat(i) > max) {
                max = list.getFloat(i)
            }
        }
        for (i in 0 until list.length()) {
            if (list.getFloat(i) == max) {
                maxKeys.add(key)
            }

            key++
        }

        if (maxKeys.count() > 0) {
            return maxKeys.random()
        }

        return 0
    }

    fun isValidMove(validMoves: List<Int>, move: Int): Boolean {
        var found = false
        for (j in 0 until validMoves.count()) {
            if (move == validMoves[j]) {
                found = true
                break
            }
        }

        return found
    }

    fun getStats(model: MultiLayerNetwork, games: Int): Int = runBlocking {
        val jobs = mutableListOf<Job>()

        var wins = 0
        var losses = 0
        var draws = 0
        var invalid = 0
        var aiRole = 1f

        for (i in 0..games - 1) {
            jobs += launch {
                var floatBoard = arrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
                var turn = 1f
                outer@ while (true) {
                    var validMoves = getMoves(floatBoard)
                    var move: Int

                    if (turn != aiRole) {
                        // move = ArgMax(Board.GetMoveRewards(floatBoard))
                        move = validMoves.random()
                    } else {
                        var predicitons = model.output(
                                Nd4j.createFromArray(arrayOf(floatBoard))
                        )
                        move = argMax(predicitons)
                    }

                    if (!isValidMove(validMoves, move)) {
                        invalid++
                        break@outer
                    }

                    floatBoard[move] = turn

                    var moves = getMoves(floatBoard)
                    var state = getBoardState(floatBoard)
                    if (state == 0f && moves.count() == 0) {
                        draws++
                        break
                    } else if (state == 1f) {
                        if (state == aiRole)
                            wins++
                        else
                            losses++
                        break
                    } else if (state == 2f) {
                        if (state == aiRole)
                            wins++
                        else
                            losses++
                        break
                    }

                    if (turn == 1f)
                        turn = 2f
                    else
                        turn = 1f
                }

                if (aiRole == 1f)
                    aiRole = 2f
                else
                    aiRole = 1f
            }
        }

        jobs.forEach { it.join() }

        val lossesAndInvalid = losses + invalid
        println("Wins: $wins, Losses: $lossesAndInvalid, Draws: $draws, Invalids: $invalid")

        return@runBlocking lossesAndInvalid
    }

    fun reward(boardRewards: MutableList<Array<Float>>, min: Float, max: Float): MutableList<Array<Float>> {
        for (j in 0 until boardRewards.count()) {
            for (k in 0..8) {
                if (boardRewards[j][k] == -1f) {
                    var reward = min + ((max - min) / 2)
                    if (j == boardRewards.count() - 1)
                        reward = max
                    boardRewards[j][k] = reward
                }
            }
        }

        return boardRewards
    }

    fun rewardReverse(boardRewards: MutableList<Array<Float>>, min: Float, max: Float): MutableList<Array<Float>> {
        for (j in boardRewards.count() - 1 until -1) {
            for (k in 0..8) {
                if (boardRewards[j][k] == -1f) {
                    var reward = max - ((max - min) / 2)
                    if (j == 0)
                        reward = min
                    boardRewards[j][k] = reward
                }
            }
        }

        return boardRewards
    }

    fun createModel(): MultiLayerNetwork {
        val multiLayerConfiguration = NeuralNetConfiguration.Builder()
                .seed(1)
                .updater(Adam())
                .list()
                .layer(
                        DenseLayer.Builder()
                                .nIn(9)
                                .nOut(180)
                                .activation(Activation.SWISH)
                                .build()
                )
                .layer(
                        DenseLayer.Builder()
                                .nOut(90)
                                .activation(Activation.SWISH)
                                .build()
                )
                .layer(
                        OutputLayer.Builder()
                                .nOut(9)
                                .activation(Activation.SWISH)
                                .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                                .build()
                )
                .build()

        val multiLayerNetwork = MultiLayerNetwork(multiLayerConfiguration)
        multiLayerNetwork.init()
        return multiLayerNetwork
    }

    @JvmStatic
    fun main(args: Array<String>) = runBlocking {
        var gammaStart = 1000f
        var iterations = 500f
        var iteration = 0f
        var gamesPerSession = 100
        var aiRole = 1f
        var gamma: Int
        var validMoveReward = 0f
        var drawReward = 0.9f
        var winReward = 1f
        var loseReward = 0f
        var epochs = 30

        var model = createModel()

        println(model.summary())
        println("Iterations: $iterations")
        println("Games per session: $gamesPerSession")
        println("Epochs: $epochs")
        println("Valid move reward: $validMoveReward")
        println("Draw reward $drawReward")
        println("Win reward $winReward")
        println("Lose reward $loseReward")

        while (true) {
            gamma = (gammaStart / exp(iteration / (iterations / 2f))).toInt()

            var moveBoards = mutableListOf<Array<Float>>()
            var boardRewards = mutableListOf<Array<Float>>()

            val jobs = mutableListOf<Job>()

            for (i in 0..gamesPerSession) {
                jobs += launch {
                    var floatBoard = arrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
                    var turn = 1f
                    var turns = 0
                    while (true) {
                        if (turn == aiRole) {
                            moveBoards.add(floatBoard.clone())
                        }

                        var validMoves = getMoves(floatBoard)
                        var move: Int

                        if (turn != aiRole) {
                            if (Random.nextInt(gammaStart.toInt()) < gamma) {
                                move = argMax(getMoveRewards(floatBoard))
                            } else {
                                move = validMoves.random()
                            }
                        } else {
                            if (Random.nextInt(gammaStart.toInt()) < gamma) {
                                if (Random.nextInt(gammaStart.toInt()) < gamma) {
                                    move = validMoves.random()
                                } else {
                                    move = argMax(getMoveRewards(floatBoard))
                                }
                            } else {
                                if (turns == 0) {
                                    move = validMoves.random()
                                } else {
                                    var predicitons = model.output(
                                            Nd4j.createFromArray(arrayOf(floatBoard))
                                    )
                                    move = argMax(predicitons)
                                }
                            }
                        }

                        if (!isValidMove(validMoves, move)) {
                            print("\r$iteration: $i     ")
                            for (j in 0 until boardRewards.count()) {
                                for (k in 0..8) {
                                    if (boardRewards[j][k] == -1f) {
                                        boardRewards[j][k] = validMoveReward
                                    }
                                }
                            }

                            var rewards = arrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
                            validMoves.forEach {
                                rewards[it] = validMoveReward
                            }

                            boardRewards.add(rewards)
                            break
                        }

                        floatBoard[move] = turn
                        var moves = getMoves(floatBoard)
                        var state = getBoardState(floatBoard)
                        if (state == 0f && moves.count() == 0) {
                            print("\r$iteration: $i     ")
                            if (turn == aiRole) {
                                var rewards = arrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
                                validMoves.forEach {
                                    rewards[it] = validMoveReward
                                }

                                rewards[move] = -1f

                                boardRewards.add(rewards)
                            }

                            boardRewards = reward(boardRewards, validMoveReward, drawReward)

                            break
                        } else if (state == 1f) {
                            print("\r$iteration: $i     ")
                            if (turn == aiRole) {
                                var rewards = arrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
                                validMoves.forEach {
                                    rewards[it] = validMoveReward
                                }

                                rewards[move] = -1f

                                boardRewards.add(rewards)
                            }

                            if (state == aiRole)
                                boardRewards = reward(boardRewards, validMoveReward, winReward)
                            else
                                boardRewards = rewardReverse(boardRewards, loseReward, validMoveReward)


                            break
                        } else if (state == 2f) {
                            if (turn == aiRole) {
                                var rewards = arrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
                                validMoves.forEach {
                                    rewards[it] = validMoveReward
                                }

                                rewards[move] = -1f

                                boardRewards.add(rewards)
                            }

                            print("\r$iteration: $i     ")

                            if (state == aiRole)
                                boardRewards = reward(boardRewards, validMoveReward, winReward)
                            else
                                boardRewards = rewardReverse(boardRewards, loseReward, validMoveReward)

                            break
                        } else {

                            if (turn == aiRole) {
                                var rewards = arrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
                                validMoves.forEach {
                                    rewards[it] = validMoveReward
                                }

                                rewards[move] = -1f

                                boardRewards.add(rewards)
                            }

                        }

                        if (turn == 1f)
                            turn = 2f
                        else
                            turn = 1f

                        turns++
                        if (turns > 10) {
                            break
                        }
                    }

                    if (aiRole == 1f)
                        aiRole = 2f
                    else
                        aiRole = 1f
                }
            }

            jobs.forEach { it.join() }

            for (epoch in 0..epochs) {
                model.fit(
                        Nd4j.createFromArray(moveBoards.toTypedArray()),
                        Nd4j.createFromArray(boardRewards.toTypedArray())
                )
            }

            if (iteration % 50f == 0f) {

                if (getStats(model, 100) == 0) {
                    if (getStats(model, 1000) == 0) {
                        getStats(model, 10000)
                        return@runBlocking
                    }
                }
            }

            if (iteration >= iterations) {
                break
            }
            iteration++
        }
    }

    fun playRandomAgainstBlankModelCoroutines(args: Array<String>) {
        val model = createModel()
        val start = System.currentTimeMillis()
        getStats(model, 1000)
        println((System.currentTimeMillis() - start))
    }

    fun playRandomAgainstAlgo(args: Array<String>) {
        var board = arrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        var turn = 1f
        while (true) {
            printBoard(board)
            val state = getBoardState(board)
            if (state == 1f) {
                println("Naughts win")
                break
            } else if (state == 2f) {
                println("Crosses win")
                break
            } else if (state == 0f) {
                val moves = getMoves(board)
                if (turn == 2f) {
                    val move = argMax(getMoveRewards(board))
                    board[move] = turn
                } else {
                    if (moves.count() > 0) {
                        val move = moves.random()
                        board[move] = turn
                    } else {
                        println("Draw")
                        break
                    }
                }
                if (turn == 1f)
                    turn = 2f
                else
                    turn = 1f
            }
        }
    }

    fun playRandomGame(args: Array<String>) {
        var board = arrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        var turn = 1f
        while (true) {
            printBoard(board)
            val state = getBoardState(board)
            if (state == 1f) {
                println("Naughts win")
                break
            } else if (state == 2f) {
                println("Crosses win")
                break
            } else if (state == 0f) {
                val moves = getMoves(board)
                if (moves.count() > 0) {
                    val move = moves.random()
                    board[move] = turn
                    if (turn == 1f)
                        turn = 2f
                    else
                        turn = 1f
                } else {
                    println("Draw")
                    break
                }
            }
        }
    }

    fun mnist(args: Array<String>) {
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

        for (i in 0 until 10) {
            GlobalScope.launch {
                computeNotSuspend(i, model)
            }
        }

        Thread.sleep(5000)
    }
}