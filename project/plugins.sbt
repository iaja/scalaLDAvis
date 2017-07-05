logLevel := Level.Warn
resolvers += Classpaths.sbtPluginReleases
resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"


addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.3")
addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.2.6")


addSbtPlugin("com.eed3si9n"       %   "sbt-unidoc"              % "0.3.2")
addSbtPlugin("com.github.gseitz"  %   "sbt-release"             % "1.0.5")
addSbtPlugin("com.jsuereth"       %   "sbt-pgp"                 % "1.0.0")
addSbtPlugin("org.scoverage"      %   "sbt-scoverage"           % "1.5.0")
addSbtPlugin("com.timushev.sbt"   %   "sbt-updates"             % "0.1.9")
addSbtPlugin("com.typesafe.sbt"   %   "sbt-ghpages"             % "0.5.4")
addSbtPlugin("com.updateimpact"   %   "updateimpact-sbt-plugin" % "2.1.1")
addSbtPlugin("org.xerial.sbt"     %   "sbt-sonatype"            % "1.1")
addSbtPlugin("com.codacy"         %   "sbt-codacy-coverage"     % "1.3.8")