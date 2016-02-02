lazy val dl4jVersion = settingKey[String]("dl4j-version")

lazy val root = project.in(file(".")).settings(
  name := "CNN MNIST Example",
  version := "0.1",
  scalaVersion := "2.11.7", //Watch out: deeplearning4j-scaleout module depends on 2.10 Akka.
  dl4jVersion := "0.4-rc3.8",
  libraryDependencies ++= Seq(
    "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion.value,
    "org.nd4j" % "nd4j-x86" % dl4jVersion.value,
    "org.nd4j" %% "nd4s" % dl4jVersion.value
  ),
  scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature", "-language:implicitConversions", "-language:postfixOps"),
  initialCommands in console := "import org.deeplearning4j.examples.demo._; import org.nd4j.linalg.factory.Nd4j; import org.nd4s.Implicits._"
)

