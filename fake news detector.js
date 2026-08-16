
## 3. `fakeNewsDetector.js`

```javascript
// Fake News Detection using Naive Bayes
// JavaScript / Node.js

class NaiveBayes {
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

        let scores = {};

        for (const currentClass of classes) {
            // Prior probability
            let score = Math.log(
                this.classCounts[currentClass] / totalDocuments
            );

            const vocabularySize = this.vocabulary.size;

            for (const word of words) {
                const count =
                    this.wordCounts[currentClass][word] || 0;

                // Laplace smoothing
                const probability =
                    (count + 1) /
                    (this.totalWords[currentClass] + vocabularySize);

                score += Math.log(probability);
            }

            scores[currentClass] = score;
        }

        return scores.REAL > scores.FAKE ? "REAL" : "FAKE";
    }
}


// --------------------------------------------------
// Training Dataset
// --------------------------------------------------

const classifier = new NaiveBayes();

const trainingData = [

    {
        text: "The government announced a new education program for students",
        label: "REAL"
    },

    {
        text: "The central bank released its official economic report",
        label: "REAL"
    },

    {
        text: "Scientists published a study about climate change",
        label: "REAL"
    },

    {
        text: "The president signed a new law after parliament approval",
        label: "REAL"
    },

    {
        text: "The health department announced a vaccination program",
        label: "REAL"
    },

    {
        text: "The university opened a new research center for science",
        label: "REAL"
    },

    {
        text: "Government officials released the annual budget report",
        label: "REAL"
    },

    {
        text: "Researchers developed a new technology after several years of study",
        label: "REAL"
    },

    {
        text: "Scientists confirm that drinking magic water makes humans immortal",
        label: "FAKE"
    },

    {
        text: "Celebrity discovers secret medicine that can cure every disease",
        label: "FAKE"
    },

    {
        text: "Aliens secretly control the government according to an unbelievable report",
        label: "FAKE"
    },

    {
        text: "Eating one special fruit gives people unlimited superpowers",
        label: "FAKE"
    },

    {
        text: "A mysterious secret machine can make everyone rich overnight",
        label: "FAKE"
    },

    {
        text: "Scientists reveal that the moon is made completely of cheese",
        label: "FAKE"
    },

    {
        text: "Drinking a magical liquid allows humans to live forever",
        label: "FAKE"
    },

    {
        text: "A secret government device can control the weather anywhere",
        label: "FAKE"
    }
];


// Train the classifier
for (const item of trainingData) {
    classifier.train(item.text, item.label);
}


// --------------------------------------------------
// Function for detecting fake news
// --------------------------------------------------

function detectNews(news) {
    const result = classifier.predict(news);

    console.log("\n----------------------------------------");
    console.log("News:", news);
    console.log("Prediction:", result);
    console.log("----------------------------------------");

    return result;
}


// --------------------------------------------------
// Example Predictions
// --------------------------------------------------

detectNews(
    "Government announces a new education program for students"
);

detectNews(
    "Scientists confirm that a magical drink makes humans immortal"
);

detectNews(
    "Researchers published a new study about climate change"
);

detectNews(
    "A secret machine can make everyone rich overnight"
);
