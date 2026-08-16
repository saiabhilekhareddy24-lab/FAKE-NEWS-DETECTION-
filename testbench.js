// Testbench for Fake News Detection

const NaiveBayes = class {
    constructor() {
        this.wordCounts = {
            REAL: {},
            FAKE: {}
        };

        this.classCounts = {
            REAL: 0,
            FAKE: 0
        };

        this.totalWords = {
            REAL: 0,
            FAKE: 0
        };

        this.vocabulary = new Set();
    }

    tokenize(text) {
        return text
            .toLowerCase()
            .replace(/[^a-zA-Z0-9\s]/g, "")
            .split(/\s+/)
            .filter(word => word.length > 2);
    }

    train(text, label) {
        const words = this.tokenize(text);

        this.classCounts[label]++;

        for (const word of words) {
            this.vocabulary.add(word);

            if (!this.wordCounts[label][word]) {
                this.wordCounts[label][word] = 0;
            }

            this.wordCounts[label][word]++;
            this.totalWords[label]++;
        }
    }

    predict(text) {
        const words = this.tokenize(text);
        const classes = ["REAL", "FAKE"];

        const totalDocuments =
            this.classCounts.REAL + this.classCounts.FAKE;

        const scores = {};

        for (const currentClass of classes) {
            let score = Math.log(
                this.classCounts[currentClass] / totalDocuments
            );

            for (const word of words) {
                const count =
                    this.wordCounts[currentClass][word] || 0;

                const probability =
                    (count + 1) /
                    (this.totalWords[currentClass] +
                        this.vocabulary.size);

                score += Math.log(probability);
            }

            scores[currentClass] = score;
        }

        return scores.REAL > scores.FAKE ? "REAL" : "FAKE";
    }
};


// Training data
const classifier = new NaiveBayes();

const trainingData = [
    ["government announces education program", "REAL"],
    ["scientists publish climate study", "REAL"],
    ["health department announces vaccination program", "REAL"],
    ["central bank releases economic report", "REAL"],
    ["government releases annual budget", "REAL"],
    ["university opens research center", "REAL"],

    ["magic drink makes humans immortal", "FAKE"],
    ["secret medicine cures every disease", "FAKE"],
    ["aliens secretly control government", "FAKE"],
    ["fruit gives humans superpowers", "FAKE"],
    ["machine makes everyone rich overnight", "FAKE"],
    ["moon is completely made of cheese", "FAKE"]
];

for (const [text, label] of trainingData) {
    classifier.train(text, label);
}


// Test cases
const testCases = [
    {
        input: "Government announces a new education program",
        expected: "REAL"
    },
    {
        input: "Scientists publish a climate research study",
        expected: "REAL"
    },
    {
        input: "Magic drink makes humans live forever",
        expected: "FAKE"
    },
    {
        input: "Secret machine makes everyone rich overnight",
        expected: "FAKE"
    },
    {
        input: "Health department announces vaccination program",
        expected: "REAL"
    },
    {
        input: "Aliens secretly control the government",
        expected: "FAKE"
    }
];


// Run testbench
console.log("========================================");
console.log("      FAKE NEWS DETECTION TESTBENCH");
console.log("========================================\n");

let correct = 0;

testCases.forEach((test, index) => {

    const prediction = classifier.predict(test.input);

    const status =
        prediction === test.expected ? "PASS" : "FAIL";

    if (prediction === test.expected) {
        correct++;
    }

    console.log(`Test ${index + 1}`);
    console.log(`Input     : ${test.input}`);
    console.log(`Expected  : ${test.expected}`);
    console.log(`Predicted : ${prediction}`);
    console.log(`Status    : ${status}`);
    console.log("----------------------------------------");
});

const accuracy = (correct / testCases.length) * 100;

console.log(`\nTest Accuracy: ${accuracy.toFixed(2)}%`);
console.log("========================================");
