// firebase.js
const admin = require('firebase-admin');
const bcrypt = require('bcrypt');
const { getStorage } = require('firebase-admin/storage');

admin.initializeApp({
  credential: admin.credential.cert({
    projectId: process.env.FIREBASE_PROJECT_ID,
    clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
    privateKey: process.env.FIREBASE_PRIVATE_KEY.replace(/\\n/g, '\n'),
  }),
  storageBucket: process.env.FIREBASE_STORAGE_BUCKET
});

const db = admin.firestore();
const storage = getStorage();

async function uploadFile(bucketPath, fileBuffer, mimeType) {
  console.log("Uploading file:", fileBuffer);
  const file = storage.bucket().file(bucketPath);
  const options = {
    metadata: {
      contentType: mimeType,
    },
  };

  if (!fileBuffer) {
    throw new Error('File buffer is undefined');
  }

  await file.save(fileBuffer, options);
  console.log(`File uploaded to ${bucketPath}`);
  return bucketPath;
}

// Completed Jobs
async function saveCompletedJob(jobId, minerId, huggingFaceRepoId) {
  const completedJobsRef = db.collection('completed_jobs');
  try {
      const newJob = {
          jobId: jobId,
          minerId: minerId,
          huggingFaceRepoId: huggingFaceRepoId,
          completedAt: admin.firestore.FieldValue.serverTimestamp()
      };
      await completedJobsRef.add(newJob);
      console.log(`Completed job saved with jobId: ${jobId}`);
  } catch (error) {
      console.error("Error saving completed job:", error);
      throw error;
  }
}

async function addTrainingJob(jobData, paramsCount, uploadedTrainingFile, uploadedValidationFile, trainingScriptFile) {
  try {
    // First, add a preliminary document to Firestore to generate docRef and get the ID
    const docRef = await db.collection('fine_tuning_jobs').add({
      ...jobData,
      paramsCount, // Add the parameters count to the Firestore document
      createdAt: admin.firestore.FieldValue.serverTimestamp(),
      status: 'pending'
    });

    // Initialize file paths using the generated docRef ID
    let trainingFilePath = uploadedTrainingFile ? `training/${docRef.id}/${uploadedTrainingFile.originalname}` : '';
    let validationFilePath = uploadedValidationFile ? `validation/${docRef.id}/${uploadedValidationFile.originalname}` : '';
    let scriptPath = trainingScriptFile ? `scripts/${docRef.id}/${new Date().toISOString().replace(/:/g, '-')}-script.py` : '';

    // Upload the files if they exist and update the paths in the document
    if (uploadedTrainingFile) {
      await uploadFile(trainingFilePath, uploadedTrainingFile.buffer, uploadedTrainingFile.mimetype);
    }
    if (uploadedValidationFile) {
      await uploadFile(validationFilePath, uploadedValidationFile.buffer, uploadedValidationFile.mimetype);
    }
    if (trainingScriptFile) {
      await uploadFile(scriptPath, trainingScriptFile.buffer, 'text/x-python-script');
    }

    // Update the Firestore document with the file paths
    await docRef.update({
      trainingFilePath,
      validationFilePath,
      scriptPath
    });

    console.log("Fine-tuning job submitted with ID:", docRef.id);
    return docRef.id;  // Returning the document ID for further use
  } catch (error) {
    console.error("Error submitting fine-tuning job:", error);
    throw error;
  }
}

// Fetch training jobs
async function fetchPendingTrainingJobs() {
  try {
    const querySnapshot = await db.collection('fine_tuning_jobs')
      .where('status', '==', 'pending')
      .get();

    if (querySnapshot.empty) {
      console.log('No pending training jobs found.');
      return [];
    }

    const pendingJobs = querySnapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));

    console.log(pendingJobs);  // Log the pending jobs to console for verification
    return pendingJobs;
  } catch (error) {
    console.error("Error fetching pending training jobs:", error);
    return [];  // Return an empty array in case of error
  }
}

async function fetchPendingJobDetails() {
  try {
    const querySnapshot = await db.collection('fine_tuning_jobs')
      .where('status', '==', 'pending')
      .get();

    if (querySnapshot.empty) {
      console.log('No pending training jobs found.');
      return [];
    }

    const pendingJobDetails = querySnapshot.docs.map(doc => ({
      id: doc.id,
      fineTuningType: doc.data().fineTuningType,
      modelId: doc.data().baseModel,
      params: doc.data().paramsCount,
      status: doc.data().status
    }));

    console.log(pendingJobDetails);  // Log the details to console for verification
    return pendingJobDetails;
  } catch (error) {
    console.error("Error fetching pending job details:", error);
    return [];  // Return an empty array in case of error
  }
}

async function fetchJobDetailsById(docId) {
  try {
    const docRef = db.collection('fine_tuning_jobs').doc(docId);
    const doc = await docRef.get();

    if (!doc.exists) {
      console.log('No such document!');
      return null;
    }

    console.log('Document data:', doc.data());
    return doc.data();
  } catch (error) {
    console.error("Error fetching document:", error);
    return null;  // Return null in case of error
  }
}

async function updatestatus(docId, status){
  const docRef = db.collection('fine_tuning_jobs').doc(docId);
  try {
    
    const doc = await docRef.get();

    if (!doc.exists) {
      console.log('No such document!');
      return null;
    }

    // Update the job status to 'running'
    await docRef.update({
      status: status
    });
  } catch (error) {
    console.error("Error updating job status:", error);
  }
}

async function start_training(docId, minerId, systemDetails) {
  const docRef = db.collection('fine_tuning_jobs').doc(docId);
  const jobExecutionRef = db.collection('job_executions');

  try {
    const doc = await docRef.get();

    if (!doc.exists) {
      console.log('No such document!');
      return null;
    }

    // Update the job status to 'running'
    await docRef.update({
      status: 'running'
    });

    // Create a job execution record
    const jobExecutionData = {
      userId: doc.data().userId, // Assuming the original job document has a userId field
      minerId: minerId,
      jobId: docId,
      systemDetails: systemDetails,
      status: 'running',
      startDate: admin.firestore.FieldValue.serverTimestamp(),
      lastUpdated: admin.firestore.FieldValue.serverTimestamp()
    };

    await jobExecutionRef.add(jobExecutionData);

    // Initialize URLs
    let scriptUrl, trainingFileUrl, validationFileUrl;

    // Retrieve file URLs if necessary
    if (doc.data().scriptPath) {
      const scriptRef = storage.bucket().file(doc.data().scriptPath);
      [scriptUrl] = await scriptRef.getSignedUrl({ action: 'read', expires: '03-09-2491' });
    }
    if (doc.data().trainingFilePath) {
      const trainingRef = storage.bucket().file(doc.data().trainingFilePath);
      [trainingFileUrl] = await trainingRef.getSignedUrl({ action: 'read', expires: '03-09-2491' });
    }
    if (doc.data().validationFilePath) {
      const validationRef = storage.bucket().file(doc.data().validationFilePath);
      [validationFileUrl] = await validationRef.getSignedUrl({ action: 'read', expires: '03-09-2491' });
    }

    const updatedData = {
      ...doc.data(),
      scriptUrl: scriptUrl || '', // Provide fallback empty string if undefined
      trainingFileUrl: trainingFileUrl || '', // Provide fallback empty string if undefined
      validationFileUrl: validationFileUrl || '' // Provide fallback empty string if undefined
    };

    return updatedData;

  } catch (error) {
    console.error("Error in starting training job:", error);
    return null;
  }
}

async function registerMiner(minerData) {
  const hashedPassword = await bcrypt.hash(minerData.password, 10); // Hash the password
  try {
    const docRef = await db.collection('miners').add({
      ethereumAddress: minerData.ethereumAddress,
      username: minerData.username,
      email: minerData.email,
      password: hashedPassword, // Store the hashed password
      createdAt: admin.firestore.FieldValue.serverTimestamp(),
    });
    console.log("Miner registered with ID:", docRef.id);
    return { username: minerData.username, password: minerData.password };
  } catch (error) {
    console.error("Error registering miner:", error);
    throw error;
  }
}

// // Function to log miner's listening status
// async function logMinerListening(minerId) {
//   const listeningRef = db.collection('listening_miners');
//   try {
//     const docRef = await listeningRef.add({
//       minerId: minerId,
//       startedListening: admin.firestore.FieldValue.serverTimestamp(),
//       status: 'listening'
//     });
//     console.log("Miner listening logged with ID:", docRef.id);
//     return docRef.id;  // Returning the document ID for further reference
//   } catch (error) {
//     console.error("Error logging miner listening:", error);
//     throw error;
//   }
// }

// Call logMinerListening when a miner successfully logs in

async function authenticateMiner(username, password) {
  const minersRef = db.collection('miners');
  const snapshot = await minersRef.where('username', '==', username).limit(1).get();
  if (snapshot.empty) {
      throw new Error('No matching user');
  }

  const userDoc = snapshot.docs[0];
  const userData = userDoc.data();
  const userId = userDoc.id;

  const passwordMatch = await bcrypt.compare(password, userData.password);
  if (passwordMatch) {
      console.log("Authenticated miner ID:", userId);
      return { userId, ...userData };
  } else {
      throw new Error('Invalid credentials');
  }
}

async function logMinerListening(minerId) {
  const listeningRef = db.collection('listening_miners');
  try {
      const docRef = await listeningRef.add({
          minerId: minerId,
          startedListening: admin.firestore.FieldValue.serverTimestamp(),
          status: 'listening'
      });
      console.log("Miner listening logged with ID:", docRef.id);
      return docRef.id;  // Returning the document ID for further reference
  } catch (error) {
      console.error("Error logging miner listening:", error);
      throw error;
  }
}

async function saveSystemDetails(minerId, systemDetails) {
  const systemDetailsRef = db.collection('miner_system_details');
  try {
      const docRef = await systemDetailsRef.add({
          minerId: minerId,
          systemDetails: systemDetails,
          loggedAt: admin.firestore.FieldValue.serverTimestamp()
      });
      console.log("System details saved with ID:", docRef.id);
      return docRef.id;  // Returning the document ID for further reference
  } catch (error) {
      console.error("Error saving system details:", error);
      throw error;
  }
}


module.exports = {
  addTrainingJob, updatestatus, saveCompletedJob, logMinerListening, saveSystemDetails, authenticateMiner, start_training, registerMiner, fetchPendingTrainingJobs, fetchPendingJobDetails, fetchJobDetailsById
};
