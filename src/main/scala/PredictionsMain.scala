import org.tensorflow.{DataType, Graph, Session, Tensor}
import java.lang
import java.nio.FloatBuffer

object PredictionsMain {
    def main(args:Array[String]): Unit = {
        val graph = new Graph()

        val aBuffer: FloatBuffer = FloatBuffer.allocate(1).put(3F)
        aBuffer.flip()
        val bBuffer: FloatBuffer = FloatBuffer.allocate(1).put(2F)
        bBuffer.flip()
        val xBuffer: FloatBuffer = FloatBuffer.allocate(1).put(1F)
        xBuffer.flip()
        val yBuffer: FloatBuffer = FloatBuffer.allocate(1).put(4F)
        yBuffer.flip()


        val a = graph.opBuilder("Const", "a")
          .setAttr("dtype", DataType.FLOAT)
          .setAttr("value", Tensor.create(Array(1L), aBuffer))
          .build

        val b = graph.opBuilder("Const", "b")
          .setAttr("dtype", DataType.FLOAT)
          .setAttr("value", Tensor.create(Array(1L), bBuffer))
          .build

        val x = graph.opBuilder("Placeholder", "x")
          .setAttr("dtype", DataType.FLOAT)
          .build
        val y = graph.opBuilder("Placeholder", "y")
          .setAttr("dtype", DataType.FLOAT)
          .build

        val ax = graph.opBuilder("Mul", "ax")
          .addInput(a.output(0))
          .addInput(x.output(0))
          .build
        val by = graph.opBuilder("Mul", "by")
          .addInput(b.output(0)).addInput(y.output(0))
          .build
        val z = graph.opBuilder("Add", "z")
          .addInput(ax.output(0))
          .addInput(by.output(0))
          .build

        val sess = new Session(graph)

        val tensor = sess.runner.fetch("z")
          .feed("x", Tensor.create(Array(1L), xBuffer))
          .feed("y", Tensor.create(Array(1L), yBuffer))
          .fetch("z")
          .run.get(0)

        val result = Array.ofDim[Float](1)
        println(tensor.copyTo(result)(0))
    }
}
