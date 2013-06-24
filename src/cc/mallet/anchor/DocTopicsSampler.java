/* Copyright (C) 2005 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.	For further
   information, see the file `LICENSE' included with this distribution. */

package cc.mallet.anchor;

import java.util.*;
import java.util.logging.Logger;
import java.util.zip.*;

import java.io.*;
import java.text.NumberFormat;

import cc.mallet.topics.*;
import cc.mallet.types.*;
import cc.mallet.util.*;


/**
 *  Estimate topics for new documents using Gibbs sampling with fixed, dense topics.
 */

public class DocTopicsSampler implements Serializable {

	private static Logger logger = MalletLogger.getLogger(DocTopicsSampler.class.getName());
	
	static cc.mallet.util.CommandOption.String inputFile = new cc.mallet.util.CommandOption.String
		(DocTopicsSampler.class, "input", "FILENAME", true, null,
		 "The filename of a mallet instance list.", null);
	
	static cc.mallet.util.CommandOption.Integer numTopicsOption = new cc.mallet.util.CommandOption.Integer
		(DocTopicsSampler.class, "num-topics", "INTEGER", true, 10,
		 "The number of topics.", null);
	
	static cc.mallet.util.CommandOption.Integer burnInOption = new cc.mallet.util.CommandOption.Integer
		(DocTopicsSampler.class, "burn-in", "INTEGER", true, 3,
		 "For each doc, sweep this number of times before saving samples.", null);
	
	static cc.mallet.util.CommandOption.Integer numSamplesOption = new cc.mallet.util.CommandOption.Integer
		(DocTopicsSampler.class, "num-samples", "INTEGER", true, 5,
		 "The number of samples to use for doc-topic proportions.", null);
	
	static cc.mallet.util.CommandOption.String topicsFile = new cc.mallet.util.CommandOption.String
		(DocTopicsSampler.class, "topics-file", "FILENAME", true, "topic-words.txt",
		 "The filename from which to read topic-word weights.", null);

	static cc.mallet.util.CommandOption.String docTopicsFile = new cc.mallet.util.CommandOption.String
		(DocTopicsSampler.class, "doc-topics", "FILENAME", true, null,
		 "The filename to write document-topic proportions.", null);
	
	static cc.mallet.util.CommandOption.String stateFile = new cc.mallet.util.CommandOption.String
		(DocTopicsSampler.class, "state-file", "FILENAME", true, null,
		 "The filename to write a gzipped Mallet Gibbs sampling state.", null);
	
	static cc.mallet.util.CommandOption.Double alphaSumOption = new cc.mallet.util.CommandOption.Double
		(DocTopicsSampler.class, "alpha", "INTEGER", true, 5.0,
		 "The sum of the document-topic hyperparameters.", null);
	
	// The number of topics requested
	protected int numTopics;

	// the alphabet for the input data
	protected Alphabet alphabet; 
	// The size of the vocabulary
	protected int numTypes;

	// Prior parameters
	protected double alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
	protected double alphaSum;
	
	// Statistics needed for sampling.
	protected double[][] typeTopicWeights; // indexed by <feature index, topic index>
	protected double[] topicSums;

	protected Randoms random;
	
	public DocTopicsSampler (int numberOfTopics) {
		this (numberOfTopics, numberOfTopics);
	}
	
	public DocTopicsSampler (int numberOfTopics, double alphaSum) {
		this (numberOfTopics, alphaSum, new Randoms());
	}
	
	public DocTopicsSampler (int numTopics, double alphaSum, Randoms random) {

		this.numTopics = numTopics;

		this.alphaSum = alphaSum;
		this.alpha = alphaSum / numTopics;
		this.random = random;
		
		topicSums = new double[numTopics];
		
		logger.info("Sampler for local assignments: " + numTopics + " topics");
	}
	
	public Alphabet getAlphabet() { return alphabet; }
	public int getNumTopics() { return numTopics; }
	
	public void setRandomSeed(int seed) {
		random = new Randoms(seed);
	}
	
	public double[][] getTypeTopicWeights() { return typeTopicWeights; }
	public double[] getTopicTotals() { return topicSums; }

	public void readTopics(Alphabet alphabet, File topicWordsFile) throws IOException {
		this.alphabet = alphabet;
		this.numTypes = alphabet.size();
		typeTopicWeights = new double[numTypes][numTopics];

		BufferedReader in = new BufferedReader(new FileReader(topicWordsFile));
		int wordID = 0;
		String line;
		while ((line = in.readLine()) != null) {
			String[] topicValues = line.split("\\s+");
                        
			if (topicValues[0].indexOf(":") != -1) {
				for (String pair: topicValues) {
					String[] fields = pair.split(":");
					int topicID = Integer.parseInt(fields[0]);
					typeTopicWeights[wordID][topicID] = Double.parseDouble(fields[1]);
					topicSums[topicID] += typeTopicWeights[wordID][topicID];
				}
			}
			else {
				for (int topicID = 0; topicID < numTopics; topicID++) {
					typeTopicWeights[wordID][topicID] = Double.parseDouble(topicValues[topicID]);
                    topicSums[topicID] += typeTopicWeights[wordID][topicID];
				}
			}
			wordID++;
		}
		in.close();
	}

	public void setTopics(Alphabet alphabet, double[][] typeTopicWeights) {
		this.alphabet = alphabet;
		this.numTypes = alphabet.size();
		this.typeTopicWeights = typeTopicWeights;
	}

	public void writeTopics(File docTopicsFile, InstanceList instances, int burnIn, int samples) throws IOException {
		PrintWriter out = new PrintWriter(docTopicsFile);
		
		for (int doc = 0; doc < instances.size(); doc++) {
			FeatureSequence tokenSequence =
				(FeatureSequence) instances.get(doc).getData();
			int length = tokenSequence.getLength();
			int[] topicSequence = new int[length];			

			// initialize
			sampleTopicsForOneDoc (tokenSequence, topicSequence, true);

			// sample to burn-in
			for (int iteration = 0; iteration < burnIn; iteration++) {
				sampleTopicsForOneDoc (tokenSequence, topicSequence, false);
			}

			// Now start saving values

			int[] topicSums = new int[numTopics];
			for (int iteration = 0; iteration < samples; iteration++) {
				int[] topicCounts = sampleTopicsForOneDoc (tokenSequence, topicSequence, false);

				for (int topic = 0; topic < numTopics; topic++) {
					topicSums[topic] += topicCounts[topic];
				}
			}

			double normalizer = 1.0 / (length * samples + alphaSum);
			Formatter line = new Formatter();
			line.format("%d\t%s", doc, instances.get(doc).getName());
			for (int topic = 0; topic < numTopics; topic++) {
				line.format("\t%f", (alpha + topicSums[topic]) * normalizer);
			}
			out.println(line);

			if (doc % 1000 == 0) { System.out.format("Processing document %d\n", doc); }
		}

		out.close();
	}
	
	protected int[] sampleTopicsForOneDoc (FeatureSequence tokenSequence,
										   int[] topicSequence, boolean initializing) {

		double[] currentTypeTopicWeights;
		int type, oldTopic, newTopic;
		double topicWeightsSum;
		int docLength = tokenSequence.getLength();

		int[] localTopicCounts = new int[numTopics];

		if (! initializing) {
			//		populate topic counts
			for (int position = 0; position < docLength; position++) {
				localTopicCounts[topicSequence[position]]++;
			}
		}

		double score, sum;
		double[] topicTermScores = new double[numTopics];

		//	Iterate over the positions (words) in the document 
		for (int position = 0; position < docLength; position++) {
			type = tokenSequence.getIndexAtPosition(position);
			oldTopic = topicSequence[position];

			// Grab the relevant row from our two-dimensional array
			currentTypeTopicWeights = typeTopicWeights[type];
			
			if (! initializing) {
				//	Remove this token from all counts. 
				localTopicCounts[oldTopic]--;
			}

			// Now calculate and add up the scores for each topic for this word
			sum = 0.0;
			
			// Here's where the math happens! Note that overall performance is 
			//  dominated by what you do in this loop.
			for (int topic = 0; topic < numTopics; topic++) {
				score =
					(alpha + localTopicCounts[topic]) *
					(currentTypeTopicWeights[topic] / topicSums[topic]);
				sum += score;
				topicTermScores[topic] = score;
			}
			
			// Choose a random point between 0 and the sum of all topic scores
			double sample = random.nextUniform() * sum;

			// Figure out which topic contains that point
			newTopic = -1;
			while (sample > 0.0) {
				newTopic++;
				sample -= topicTermScores[newTopic];
			}

			// Make sure we actually sampled a topic
			if (newTopic == -1) {
				throw new IllegalStateException ("DocTopicsSampler: New topic not sampled.");
			}

			// Put that new topic into the counts
			topicSequence[position] = newTopic;
			localTopicCounts[newTopic]++;
		}
		
		return localTopicCounts;
	}
	
	public void printState (File f, InstanceList instances, int burnIn) throws IOException {
		PrintStream out =
			new PrintStream(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(f))));
		printState(out, instances, burnIn);
		out.close();
	}
	
	public void printState (PrintStream out, InstanceList instances, int burnIn) {

		out.println ("#doc source pos typeindex type topic");

		for (int doc = 0; doc < instances.size(); doc++) {
			FeatureSequence tokenSequence =
				(FeatureSequence) instances.get(doc).getData();
			int length = tokenSequence.getLength();
			int[] topicSequence = new int[length];

			// initialize
			sampleTopicsForOneDoc (tokenSequence, topicSequence, true);

			for (int iteration = 0; iteration < burnIn; iteration++) {
				sampleTopicsForOneDoc (tokenSequence, topicSequence, false);
			}

			String source = "NA";
			if (instances.get(doc).getSource() != null) {
				source = instances.get(doc).getSource().toString();
			}

			for (int position = 0; position < length; position++) {
				int type = tokenSequence.getIndexAtPosition(position);
				int topic = topicSequence[position];
				Formatter line = new Formatter();
				line.format("%d %s %d %d %s %d", doc, source, position, type, alphabet.lookupObject(type), topic);
				out.println(line);
			}
		}
	}
	
	public static void main (String[] args) throws IOException {
		CommandOption.setSummary (DocTopicsSampler.class, "Find topic distributions for documents given previously learned topics.");
		CommandOption.process (DocTopicsSampler.class, args);

		InstanceList instances = InstanceList.load (new File(inputFile.value));

		int numTopics = numTopicsOption.value;

		DocTopicsSampler lda = new DocTopicsSampler (numTopics, alphaSumOption.value);
		lda.readTopics(instances.getDataAlphabet(), new File(topicsFile.value));

		if (docTopicsFile.value != null) {
			lda.writeTopics(new File(docTopicsFile.value), instances, burnInOption.value, numSamplesOption.value);
		}

		if (stateFile.value != null) {
			lda.printState(new File(stateFile.value), instances, burnInOption.value);
		}
	}
	
}
