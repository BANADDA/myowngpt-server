require('dotenv').config();
const amqp = require('amqplib');
const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const { expressjwt: expressJwt } = require('express-jwt');
const { 
    addTrainingJob, 
    logMinerListening, 
    saveSystemDetails,
    saveCompletedJob, 
    fetchJobDetailsById, 
    registerMiner, 
    authenticateMiner, 
    fetchPendingJobDetails, 
    start_training, 
    updatestatus 
} = require('./firebase');
const { exec } = require('child_process');
const http = require('http');
const socketIo = require('socket.io');
const bodyParser = require('body-parser');
const app = express();
const server = http.createServer(app);
const io = socketIo(server);
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(bodyParser.json());

// Set up multer for file storage
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Middleware to validate JWT
const checkJwt = expressJwt({
    secret: process.env.JWT_SECRET,
    algorithms: ['HS256'],
    requestProperty: 'user' // ensures decoded token is attached to req.user
});

app.get('/', (req, res) => {
    console.log("Hello server is live...");
    res.send('Hello World');
});

// Completed Jobs
app.post('/complete-training', async (req, res) => {
    const { jobId, huggingFaceRepoId, minerId } = req.body;

    if (!jobId || !huggingFaceRepoId) {
        return res.status(400).json({ error: 'jobId and huggingFaceRepoId are required.' });
    }

    try {
        await saveCompletedJob(jobId, minerId, huggingFaceRepoId);
        res.status(200).json({ message: 'Completed job saved successfully.' });
    } catch (error) {
        console.error('Failed to save completed job:', error);
        res.status(500).json({ error: 'Failed to save completed job.' });
    }
});

// Endpoint to deploy a model
app.post('/deploy-model', async (req, res) => {
    const { model_id, model_name } = req.body;

    const process = exec(`python inference/runpod/deploy_pod.py ${model_id} ${model_name}`);

    process.stdout.on('data', (data) => {
        res.write(data); // Stream the log messages to the client
    });

    process.stderr.on('data', (data) => {
        console.error(`Deployment error: ${data}`);
        res.write(`Deployment error: ${data}`); // Stream the error messages to the client
    });

    process.on('close', (code) => {
        if (code !== 0) {
            res.status(500).end('Deployment process exited with code ' + code);
        } else {
            res.end();
        }
    });
});

// Endpoint to perform inference
app.post('/inference', async (req, res) => {
    const { endpoint_url, prompt } = req.body;

    console.log(`Received endpoint_url: ${endpoint_url}, prompt: ${prompt}`);

    if (!endpoint_url || !prompt) {
        return res.status(400).json({ error: 'endpoint_url and prompt are required.' });
    }

    exec(`python inference/runpod/inference.py ${endpoint_url} "${prompt}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error making inference: ${error.message}`);
            return res.status(500).send({ error: `Failed to make inference: ${error.message}` });
        }
        if (stderr) {
            console.error(`Inference error: ${stderr}`);
            return res.status(500).send({ error: `Inference error: ${stderr}` });
        }
        console.log(`Inference output: ${stdout}`);
        try {
            const data = JSON.parse(stdout);
            res.status(200).json(data);
        } catch (err) {
            console.error(`Failed to parse inference output: ${err.message}`);
            res.status(500).send({ error: `Failed to parse inference output: ${err.message}` });
        }
    });
});

app.get('/wandb-data', async (req, res) => {
    console.log('Endpoint /wandb-data reached');
    exec('python wandb/test.py', (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing script: ${error.message}`);
            res.status(500).send({ error: 'Failed to execute script' });
            return;
        }
        if (stderr) {
            console.error(`Script error: ${stderr}`);
            res.status(500).send({ error: 'Script error' });
            return;
        }

        // Log the stdout to check the script's output
        console.log(`Script output: ${stdout}`);

        // Send the output from the Python script as JSON
        try {
            const data = JSON.parse(stdout);
            console.log('Parsed data:', data);
            res.status(200).json(data);
        } catch (err) {
            console.error(`Failed to parse script output: ${err.message}`);
            res.status(500).send({ error: 'Failed to parse script output' });
        }
    });
});

// Route to handle the training job submissions
app.post('/submit-training', upload.fields([{ name: 'trainingFile' }, { name: 'validationFile' }]), async (req, res) => {
    const { body, files } = req;

    // Generate training script content dynamically based on request
    const scriptContent = generateTrainingScript(body);
    const scriptBuffer = Buffer.from(scriptContent, 'utf-8');
    const scriptFilename = `training_script-${Date.now()}.py`;

    try {
        // Prepare file data for upload
        const trainingFileData = files['trainingFile'] ? {
            originalname: files['trainingFile'][0].originalname,
            buffer: files['trainingFile'][0].buffer,
            mimetype: files['trainingFile'][0].mimetype
        } : null;

        const validationFileData = files['validationFile'] ? {
            originalname: files['validationFile'][0].originalname,
            buffer: files['validationFile'][0].buffer,
            mimetype: files['validationFile'][0].mimetype
        } : null;

        const scriptFileData = {
            originalname: scriptFilename,
            buffer: scriptBuffer,
            mimetype: 'text/x-python-script'
        };

        // Submit training job along with files
        const jobId = await addTrainingJob(body, trainingFileData, validationFileData, scriptFileData);
        res.status(200).send({ message: 'Training job submitted successfully!', jobId: jobId });
    } catch (error) {
        console.error('Error submitting training job:', error);
        res.status(500).send({ error: 'Failed to submit training job.' });
    }
});

app.post('/start-training/:docId', checkJwt, async (req, res) => {
    console.log('Miner ID:', req.user.minerId); // Assuming minerId is in req.user
    console.log("Request to start training received with docId:", req.params.docId);
    console.log("Request body:", req.body);

    const { docId } = req.params;
    const systemDetails = req.body;
    const minerId = req.user.minerId; // Extract minerId from the decoded JWT

    try {
        const jobDetails = await start_training(docId, minerId, systemDetails); // Pass minerId here
        if (!jobDetails) {
            res.status(404).send({ message: 'Job not found or failed to start' });
            return;
        }
        res.status(200).json(jobDetails);
    } catch (error) {
        console.error('Failed to start training job:', error);
        res.status(500).send({ error: 'Failed to start training job' });
    }
});

// Endpoint to update the status of a specific training job
app.patch('/update-status/:docId', checkJwt, async (req, res) => {
    const { docId } = req.params;
    const { status } = req.body;

    if (!status) {
        return res.status(400).json({ error: 'Status is required.' });
    }

    try {
        // Assuming `updateStatus` is a function that updates the job's status in the database
        await updatestatus(docId, status);
        res.json({ message: `Status updated to ${status} for job ${docId}` });
    } catch (error) {
        console.error('Failed to update job status:', error);
        res.status(500).json({ error: 'Failed to update job status' });
    }
});

app.get('/pending-jobs', checkJwt, async (req, res) => {
    try {
        const jobs = await fetchPendingJobDetails();
        res.status(200).json(jobs);
    } catch (error) {
        console.error('Failed to fetch pending jobs:', error);
        res.status(500).send({ error: 'Failed to fetch pending jobs' });
    }
});

app.get('/job-details/:docId', checkJwt, async (req, res) => {
    const { docId } = req.params;

    try {
        const jobDetails = await fetchJobDetailsById(docId);
        if (!jobDetails) {
            res.status(404).send({ message: 'Job not found' });
            return;
        }
        res.status(200).json(jobDetails);
    } catch (error) {
        console.error('Failed to fetch job details:', error);
        res.status(500).send({ error: 'Failed to fetch job details' });
    }
});

app.post('/login', async (req, res) => {
    const { username, password, system_details } = req.body;

    try {
        const user = await authenticateMiner(username, password);
        if (user) {
            // Log that the miner is listening
            const listen = await logMinerListening(user.userId);
            console.log("Listening: ", listen);
            
            // Save system details
            const systemDetailsId = await saveSystemDetails(user.userId, system_details);
            console.log("System details saved with ID:", systemDetailsId);

            // Ensure minerId is included properly
            const token = jwt.sign({
                username: user.username,
                minerId: user.userId
            }, process.env.JWT_SECRET, { expiresIn: '24h' });

            res.status(200).send({
                token,
                userId: user.username,
                minerId: user.userId
            });
        } else {
            res.status(401).send({ error: 'Authentication failed.' });
        }
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).send({ error: 'Login failed due to server error.' });
    }
});

// app.post('/login', async (req, res) => {
//     const { username, password, system_details } = req.body;

//     try {
//         const user = await authenticateMiner(username, password);
//         if (user) {
//             // Log that the miner is listening
//             const listen = await logMinerListening(user.userId);
//             console.log("Listening: ", listen);
//             // Ensure minerId is included properly
//             const token = jwt.sign({
//                 username: user.username,
//                 minerId: user.userId
//             }, process.env.JWT_SECRET, { expiresIn: '24h' });

//             // Save system details in Firestore
//             const minerRef = db.collection('miners').doc(user.userId);
//             await minerRef.set({ system_details }, { merge: true });

//             res.status(200).send({
//                 token,
//                 userId: user.username,
//                 minerId: user.userId
//             });
//         } else {
//             res.status(401).send({ error: 'Authentication failed.' });
//         }
//     } catch (error) {
//         console.error('Login error:', error);
//         res.status(500).send({ error: 'Login failed due to server error.' });
//     }
// });

app.post('/register-miner', async (req, res) => {
    const { ethereumAddress, username, email } = req.body;
    const password = generatePassword(); // Automatically generate a password

    try {
        const result = await registerMiner({ ethereumAddress, username, email, password });
        res.status(200).send(result); // Send back the username and generated password
    } catch (error) {
        console.error('Error registering miner:', error);
        res.status(500).send({ error: 'Failed to register miner.' });
    }
});

// Route not found (404)
app.use((req, res, next) => {
    res.status(404).send({ error: 'Not Found' });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send({ error: 'Something broke!' });
});

function generatePassword() {
    const length = 6;
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let retVal = "";
    for (let i = 0, n = charset.length; i < length; ++i) {
        retVal += charset.charAt(Math.floor(Math.random() * n));
    }
    return retVal;
}

function generateTrainingScript(trainingData) {
    const { baseModel = 'gpt2', batchSize = 8, learningRateMultiplier = 5e-5, numberOfEpochs = 3, fineTuningType = 'text-generation', huggingFaceId = 'default-dataset', suffix = 'default', seed = '42' } = trainingData;
    return `
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import random
import torch

random.seed(${parseInt(seed, 10)})
torch.manual_seed(${parseInt(seed, 10)})

model = AutoModelForCausalLM.from_pretrained("${baseModel}")
dataset = load_dataset("${fineTuningType}", "${huggingFaceId}")

training_args = TrainingArguments(
    output_dir='./results-${suffix}',
    evaluation_strategy="epoch",
    learning_rate=${parseFloat(learningRateMultiplier)},
    per_device_train_batch_size=${parseInt(batchSize, 10)},
    num_train_epochs=${parseInt(numberOfEpochs, 10)},
    save_strategy="epoch",
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)

trainer.train()
model.save_pretrained('./final_model-${suffix}')
`;
}

server.listen(port, () => {
  console.log(`Server running on port ${port}`);
});


// require('dotenv').config();
// const amqp = require('amqplib');
// const express = require('express');
// const multer = require('multer');
// const fs = require('fs');
// const path = require('path');
// const cors = require('cors');
// const jwt = require('jsonwebtoken');
// const bcrypt = require('bcrypt');
// const { expressjwt: expressJwt } = require('express-jwt');
// const { addTrainingJob, logMinerListening, saveCompletedJob, fetchJobDetailsById, registerMiner, authenticateMiner, fetchPendingJobDetails, start_training, updatestatus } = require('./firebase');
// const { exec } = require('child_process');
// const http = require('http');
// const socketIo = require('socket.io');
// const bodyParser = require('body-parser');
// const app = express();
// const server = http.createServer(app);
// const io = socketIo(server);
// const port = process.env.PORT || 3000;

// app.use(cors());
// app.use(express.urlencoded({ extended: true }));
// app.use(express.json());
// app.use(bodyParser.json());

// // Set up multer for file storage
// const storage = multer.memoryStorage();
// const upload = multer({ storage: storage });

// // Middleware to validate JWT
// const checkJwt = expressJwt({
//     secret: process.env.JWT_SECRET,
//     algorithms: ['HS256'],
//     requestProperty: 'user' // ensures decoded token is attached to req.user
// });

// // Completed Jobs
// app.post('/complete-training', async (req, res) => {
//     const { jobId, huggingFaceRepoId, minerId } = req.body;

//     if (!jobId || !huggingFaceRepoId) {
//         return res.status(400).json({ error: 'jobId and huggingFaceRepoId are required.' });
//     }

//     try {
//         await saveCompletedJob(jobId, minerId, huggingFaceRepoId);
//         res.status(200).json({ message: 'Completed job saved successfully.' });
//     } catch (error) {
//         console.error('Failed to save completed job:', error);
//         res.status(500).json({ error: 'Failed to save completed job.' });
//     }
// });


// // Endpoint to deploy a model
// app.post('/deploy-model', (req, res) => {
//     const { model_id, model_name } = req.body;

//     const process = exec(`python inference/runpod/deploy_pod.py ${model_id} ${model_name}`);

//     process.stdout.on('data', (data) => {
//         res.write(data); // Stream the log messages to the client
//     });

//     process.stderr.on('data', (data) => {
//         console.error(`Deployment error: ${data}`);
//         res.write(`Deployment error: ${data}`); // Stream the error messages to the client
//     });

//     process.on('close', (code) => {
//         if (code !== 0) {
//             res.status(500).end('Deployment process exited with code ' + code);
//         } else {
//             res.end();
//         }
//     });
// });

// // Endpoint to perform inference
// app.post('/inference', async (req, res) => {
//     const { endpoint_url, prompt } = req.body;

//     exec(`python inference/runpod/inference.py ${endpoint_url} "${prompt}"`, (error, stdout, stderr) => {
//         if (error) {
//             console.error(`Error making inference: ${error.message}`);
//             res.status(500).send({ error: 'Failed to make inference' });
//             return;
//         }
//         if (stderr) {
//             console.error(`Inference error: ${stderr}`);
//             res.status(500).send({ error: 'Inference error' });
//             return;
//         }
//         console.log(`Inference output: ${stdout}`);
//         try {
//             const data = JSON.parse(stdout);
//             res.status(200).json(data);
//         } catch (err) {
//             console.error(`Failed to parse inference output: ${err.message}`);
//             res.status(500).send({ error: 'Failed to parse inference output' });
//         }
//     });
// });

// app.get('/wandb-data', (req, res) => {
//     console.log('Endpoint /wandb-data reached');
//     exec('python wandb/test.py', (error, stdout, stderr) => {
//         if (error) {
//             console.error(`Error executing script: ${error.message}`);
//             res.status(500).send({ error: 'Failed to execute script' });
//             return;
//         }
//         if (stderr) {
//             console.error(`Script error: ${stderr}`);
//             res.status(500).send({ error: 'Script error' });
//             return;
//         }

//         // Log the stdout to check the script's output
//         console.log(`Script output: ${stdout}`);

//         // Send the output from the Python script as JSON
//         try {
//             const data = JSON.parse(stdout);
//             console.log('Parsed data:', data);
//             res.status(200).json(data);
//         } catch (err) {
//             console.error(`Failed to parse script output: ${err.message}`);
//             res.status(500).send({ error: 'Failed to parse script output' });
//         }
//     });
// });

// // Route to handle the training job submissions
// app.post('/submit-training', upload.fields([{ name: 'trainingFile' }, { name: 'validationFile' }]), async (req, res) => {
//     const { body, files } = req;

//     // Generate training script content dynamically based on request
//     const scriptContent = generateTrainingScript(body);
//     const scriptBuffer = Buffer.from(scriptContent, 'utf-8');
//     const scriptFilename = `training_script-${Date.now()}.py`;

//     try {
//         // Prepare file data for upload
//         const trainingFileData = files['trainingFile'] ? {
//             originalname: files['trainingFile'][0].originalname,
//             buffer: files['trainingFile'][0].buffer,
//             mimetype: files['trainingFile'][0].mimetype
//         } : null;

//         const validationFileData = files['validationFile'] ? {
//             originalname: files['validationFile'][0].originalname,
//             buffer: files['validationFile'][0].buffer,
//             mimetype: files['validationFile'][0].mimetype
//         } : null;

//         const scriptFileData = {
//             originalname: scriptFilename,
//             buffer: scriptBuffer,
//             mimetype: 'text/x-python-script'
//         };

//         // const numParameters = await getModelParameters(body.baseModel, huggingFaceToken);
//         // console.log(`Number of parameters for model ${body.baseModel}: ${numParameters}`);

//         // Submit training job along with files
//         const jobId = await addTrainingJob(body, trainingFileData, validationFileData, scriptFileData);
//         res.status(200).send({ message: 'Training job submitted successfully!', jobId: jobId });
//     } catch (error) {
//         console.error('Error submitting training job:', error);
//         res.status(500).send({ error: 'Failed to submit training job.' });
//     }
// });

// app.post('/start-training/:docId', checkJwt, async (req, res) => {
//     console.log('Miner ID:', req.user.minerId); // Assuming minerId is in req.user
//     console.log("Request to start training received with docId:", req.params.docId);
//     console.log("Request body:", req.body);

//     const { docId } = req.params;
//     const systemDetails = req.body;
//     const minerId = req.user.minerId; // Extract minerId from the decoded JWT

//     try {
//         const jobDetails = await start_training(docId, minerId, systemDetails); // Pass minerId here
//         if (!jobDetails) {
//             res.status(404).send({ message: 'Job not found or failed to start' });
//             return;
//         }
//         res.status(200).json(jobDetails);
//     } catch (error) {
//         console.error('Failed to start training job:', error);
//         res.status(500).send({ error: 'Failed to start training job' });
//     }
// });

// // Endpoint to update the status of a specific training job
// app.patch('/update-status/:docId', checkJwt, async (req, res) => {
//     const { docId } = req.params;
//     const { status } = req.body;

//     if (!status) {
//         return res.status(400).json({ error: 'Status is required.' });
//     }

//     try {
//         // Assuming `updateStatus` is a function that updates the job's status in the database
//         await updatestatus(docId, status);
//         res.json({ message: `Status updated to ${status} for job ${docId}` });
//     } catch (error) {
//         console.error('Failed to update job status:', error);
//         res.status(500).json({ error: 'Failed to update job status' });
//     }
// });

// app.get('/pending-jobs', checkJwt, async (req, res) => {
//     try {
//         const jobs = await fetchPendingJobDetails();
//         res.status(200).json(jobs);
//     } catch (error) {
//         console.error('Failed to fetch pending jobs:', error);
//         res.status(500).send({ error: 'Failed to fetch pending jobs' });
//     }
// });

// app.get('/job-details/:docId', checkJwt, async (req, res) => {
//     const { docId } = req.params;

//     try {
//         const jobDetails = await fetchJobDetailsById(docId);
//         if (!jobDetails) {
//             res.status(404).send({ message: 'Job not found' });
//             return;
//         }
//         res.status(200).json(jobDetails);
//     } catch (error) {
//         console.error('Failed to fetch job details:', error);
//         res.status(500).send({ error: 'Failed to fetch job details' });
//     }
// });

// app.post('/login', async (req, res) => {
//     const { username, password } = req.body;

//     try {
//         const user = await authenticateMiner(username, password);
//         if (user) {
//             // Log that the miner is listening
//             const listen = await logMinerListening(user.userId);
//             console.log("Listening: ", listen);
//             // Ensure minerId is included properly
//             const token = jwt.sign({
//                 username: user.username,
//                 minerId: user.userId
//             }, process.env.JWT_SECRET, { expiresIn: '24h' });

//             res.status(200).send({
//                 token,
//                 userId: user.username,
//                 minerId: user.userId
//             });
//         } else {
//             res.status(401).send({ error: 'Authentication failed.' });
//         }
//     } catch (error) {
//         console.error('Login error:', error);
//         res.status(500).send({ error: 'Login failed due to server error.' });
//     }
// });

// app.post('/register-miner', async (req, res) => {
//     const { ethereumAddress, username, email } = req.body;
//     const password = generatePassword(); // Automatically generate a password

//     try {
//         const result = await registerMiner({ ethereumAddress, username, email, password });
//         res.status(200).send(result); // Send back the username and generated password
//     } catch (error) {
//         console.error('Error registering miner:', error);
//         res.status(500).send({ error: 'Failed to register miner.' });
//     }
// });

// // New route to run Python script and return its output
// // app.get('/wandb-data', (req, res) => {
// //     exec('python wandb/test.py', (error, stdout, stderr) => {
// //         if (error) {
// //             console.error(`Error executing script: ${error.message}`);
// //             res.status(500).send({ error: 'Failed to execute script' });
// //             return;
// //         }
// //         if (stderr) {
// //             console.error(`Script error: ${stderr}`);
// //             res.status(500).send({ error: 'Script error' });
// //             return;
// //         }

// //         // Parse the output from the Python script
// //         let data;
// //         try {
// //             data = JSON.parse(stdout);
// //         } catch (err) {
// //             console.error(`Failed to parse script output: ${err.message}`);
// //             res.status(500).send({ error: 'Failed to parse script output' });
// //             return;
// //         }

// //         res.status(200).json(data);
// //     });
// // });

// // Route not found (404)
// app.use((req, res, next) => {
//     res.status(404).send({ error: 'Not Found' });
// });

// // Error handling middleware
// app.use((err, req, res, next) => {
//     console.error(err.stack);
//     res.status(500).send({ error: 'Something broke!' });
// });

// function generatePassword() {
//     const length = 6;
//     const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
//     let retVal = "";
//     for (let i = 0, n = charset.length; i < length; ++i) {
//         retVal += charset.charAt(Math.floor(Math.random() * n));
//     }
//     return retVal;
// }

// function generateTrainingScript(trainingData) {
//     const { baseModel = 'gpt2', batchSize = 8, learningRateMultiplier = 5e-5, numberOfEpochs = 3, fineTuningType = 'text-generation', huggingFaceId = 'default-dataset', suffix = 'default', seed = '42' } = trainingData;
//     return `
// from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
// from datasets import load_dataset
// import random
// import torch

// random.seed(${parseInt(seed, 10)})
// torch.manual_seed(${parseInt(seed, 10)})

// model = AutoModelForCausalLM.from_pretrained("${baseModel}")
// dataset = load_dataset("${fineTuningType}", "${huggingFaceId}")

// training_args = TrainingArguments(
//     output_dir='./results-${suffix}',
//     evaluation_strategy="epoch",
//     learning_rate=${parseFloat(learningRateMultiplier)},
//     per_device_train_batch_size=${parseInt(batchSize, 10)},
//     num_train_epochs=${parseInt(numberOfEpochs, 10)},
//     save_strategy="epoch",
//     save_total_limit=1
// )

// trainer = Trainer(
//     model=model,
//     args=training_args,
//     train_dataset=dataset['train'],
//     eval_dataset=dataset['validation']
// )

// trainer.train()
// model.save_pretrained('./final_model-${suffix}')
// `;
// }

// // Real-time data fetching and Socket.io integration
// const entityName = 'ai-research-lab';
// const projectName = 'gpt_v4';
// const runId = '15v9e834'; // replace with your actual run ID

// server.listen(port, () => {
//   console.log(`Server running on port ${port}`);
// });
