plugins {
    java
    `java-library`
    `maven-publish`
}

group = "org.stonesoup"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(22))
    }
    withSourcesJar()
    withJavadocJar()
}

repositories {
    mavenCentral()
}

dependencies {
    // JUnit 5 for testing
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.10.2")
    testImplementation("org.junit.jupiter:junit-jupiter-params:5.10.2")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.10.2")
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.add("--enable-native-access=ALL-UNNAMED")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
    }
}

tasks.javadoc {
    options {
        (this as StandardJavadocDocletOptions).apply {
            addStringOption("Xdoclint:none", "-quiet")
        }
    }
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])

            pom {
                name.set("Stone Soup Java Bindings")
                description.set("Java bindings for the Stone Soup tracking framework using Project Panama FFM API")
                url.set("https://github.com/dstl/Stone-Soup")

                licenses {
                    license {
                        name.set("MIT License")
                        url.set("https://opensource.org/licenses/MIT")
                    }
                }

                developers {
                    developer {
                        name.set("Stone Soup Contributors")
                        organization.set("DSTL")
                    }
                }

                scm {
                    connection.set("scm:git:git://github.com/dstl/Stone-Soup.git")
                    developerConnection.set("scm:git:ssh://github.com/dstl/Stone-Soup.git")
                    url.set("https://github.com/dstl/Stone-Soup")
                }
            }
        }
    }
}

// Profile for Java 21 with preview features
if (JavaVersion.current() == JavaVersion.VERSION_21) {
    tasks.withType<JavaCompile>().configureEach {
        options.compilerArgs.addAll(listOf("--enable-preview"))
    }
    tasks.test {
        jvmArgs("--enable-preview")
    }
}
