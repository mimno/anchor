package cc.mallet.anchor;

import java.util.*;
import java.io.*;
import cc.mallet.util.*;
import cc.mallet.types.*;

public class SpectralLDA {

	static cc.mallet.util.CommandOption.String inputFile = new cc.mallet.util.CommandOption.String
		(SpectralLDA.class, "input", "FILENAME", true, null,
		 "The filename of a mallet instance list.", null);
	
	static cc.mallet.util.CommandOption.Integer numTopicsOption = new cc.mallet.util.CommandOption.Integer
		(SpectralLDA.class, "num-topics", "INTEGER", true, 10,
		 "The number of topics to find.", null);

	static cc.mallet.util.CommandOption.String anchorWordsFile = new cc.mallet.util.CommandOption.String
		(SpectralLDA.class, "anchors-file", "FILENAME", true, null,
		 "The filename to which to write anchor words.", null);
	
	static cc.mallet.util.CommandOption.String topicsFile = new cc.mallet.util.CommandOption.String
		(SpectralLDA.class, "topics-file", "FILENAME", true, "topic-words.txt",
		 "The filename to which to write topic-word weights.", null);
	
	static cc.mallet.util.CommandOption.Integer randomProjections = new cc.mallet.util.CommandOption.Integer
		(SpectralLDA.class, "num-random-projections", "INTEGER", true, 0,
		 "The dimensionality of a random projection. If 0, the original data will be used.", null);
	
	static cc.mallet.util.CommandOption.Double randomProjectionSparsity = new cc.mallet.util.CommandOption.Double
		(SpectralLDA.class, "projection-density", "[0.0 - 1.0]", true, 1.0,
		 "The sparsity of the random projection: 0.1 means 10% of entries will be non-zero. If 1.0, a dense Gaussian RP will be used.", null);

	static cc.mallet.util.CommandOption.Integer minimumDocumentFrequency = new cc.mallet.util.CommandOption.Integer
		(SpectralLDA.class, "min-docs", "INTEGER", true, 10,
		 "Don't consider possible anchor words that appear in fewer than this many documents.", null);

	static cc.mallet.util.CommandOption.Integer randomSeedOption = new cc.mallet.util.CommandOption.Integer
		(SpectralLDA.class, "random-seed", "INTEGER", true, 0,
		 "The seed for a random number generator for use with RPs. The defualt option does not specify a seed.", null);

	BigramProbabilityMatrix matrix;

	// This is a good setting for the examples I've tried, but results may vary.
	double learningRate = 50.0;

	int numTopics = 0;
	int[] basisVectorIndices;
	boolean[] rowIsBasisVector;

	double[][] anchorsTimesAnchors;

	public SpectralLDA(BigramProbabilityMatrix matrix, StabilizedGS orthogonalizer) throws IOException {
		this.matrix = matrix;

		basisVectorIndices = orthogonalizer.getBasisVectorIndices();
		rowIsBasisVector = orthogonalizer.getRowIsBasisVector();
		this.numTopics = basisVectorIndices.length;

		anchorsTimesAnchors = new double[numTopics][numTopics];
		for (int row = 0; row < numTopics; row++) {
			for (int col = 0; col < numTopics; col++) {
				for (int word = 0; word < matrix.numColumns; word++) {
					anchorsTimesAnchors[ row ][ col ] +=
						matrix.weights[ basisVectorIndices[row] ][word] *
						matrix.weights[ basisVectorIndices[col] ][word];
				}
			}
		}
	}

	public double[] recover(int word) {
		//System.out.println("recovering word " + word + " / " + basisVectorIndices[0]);
		
		double[] conditionalProb = new double[numTopics];
		Arrays.fill(conditionalProb, 1.0 / numTopics);

		double[] gradient = new double[ numTopics ];
		
		double[] wordTimesAnchors = new double[numTopics];
		for (int i = 0; i < numTopics; i++) {
			for (int col = 0; col < matrix.numColumns; col++) {
				wordTimesAnchors[i] += matrix.weights[word][col] * matrix.weights[ basisVectorIndices[i] ][col];
			}
		}
		
		double previousSumSquaredError = Double.POSITIVE_INFINITY;
		double sumSquaredError = Double.POSITIVE_INFINITY;
		int iteration = 0;

		while ((Double.isInfinite(sumSquaredError) ||
				Math.abs(Math.sqrt(previousSumSquaredError) - Math.sqrt(sumSquaredError)) > 0.000001) &&
			   iteration < 500) {
			previousSumSquaredError = sumSquaredError;
			sumSquaredError = 0.0;

			// Find the gradient: -2 * (y'X - alpha'X'X)
			// We computed X'X once in the constructor, and we just calculated y'X.
			
			double maxDerivative = Double.NEGATIVE_INFINITY;
			for (int row = 0; row < numTopics; row++) {
				gradient[row] = wordTimesAnchors[row];
				for (int col = 0; col < numTopics; col++) {
					gradient[row] -= conditionalProb[col] * anchorsTimesAnchors[row][col];
				}
				sumSquaredError += gradient[row] * gradient[row];

				//System.out.format("%f ", gradient[row]);
				gradient[row] *= 2.0 * learningRate;

				if (gradient[row] > maxDerivative) { maxDerivative = gradient[row]; }
			}
			//			System.out.println();

			double sum = 0.0;
			for (int row = 0; row < numTopics; row++) {
				conditionalProb[row] *= Math.exp(gradient[row] - maxDerivative);
				assert(! Double.isNaN(conditionalProb[row])) : gradient[row] + " - " + maxDerivative;
				sum += conditionalProb[row];
			}
			//System.out.format("%f %f\n", sumSquaredError, sum);
			
			double normalizer = 1.0 / sum;
			for (int row = 0; row < numTopics; row++) {
				conditionalProb[row] *= normalizer;
				assert(! Double.isNaN(conditionalProb[row])) : gradient[row] + " - " + maxDerivative + " " + conditionalProb[row] + " * " + normalizer;
			}

			iteration++;
		}
		
		double entropy = 0.0;
		for (int topic = 0; topic < numTopics; topic++) {
			if (conditionalProb[topic] > 0.0) {
				entropy -= conditionalProb[topic] * Math.log(conditionalProb[topic]);
			}
		}

		if (word % 100 == 0) { 
			System.err.format("%d\t%d\t%f\t%f\t%f\n", word, iteration, Math.log(sumSquaredError), entropy, entropy / Math.log(numTopics));
		}

		return conditionalProb;
	}

	public double[] logSpaceRecover(int word) {
		// System.out.println("recovering word " + word + " / " + basisVectorIndices[0]);
		
		double[] conditionalProb = new double[numTopics];
		//conditionalProb[0] = 1.0;
		Arrays.fill(conditionalProb, 1.0 / numTopics);
		
		double[] wordTimesAnchors = new double[numTopics];
		for (int i = 0; i < numTopics; i++) {
			for (int col = 0; col < matrix.numColumns; col++) {
				wordTimesAnchors[i] += matrix.weights[word][col] * matrix.weights[ basisVectorIndices[i] ][col];
			}
		}
		
		double previousSumSquaredError = Double.POSITIVE_INFINITY;
		double sumSquaredError = Double.POSITIVE_INFINITY;
		int iteration = 0;
		while ((Double.isInfinite(sumSquaredError) ||
				Math.abs(Math.sqrt(previousSumSquaredError) - Math.sqrt(sumSquaredError)) > 0.000001) &&
			   iteration < 500) {
			previousSumSquaredError = sumSquaredError;
			sumSquaredError = 0.0;

			// Find the gradient: -2 * (y'X - alpha'X'X)
			// We computed X'X once in the constructor, and we just calculated y'X.
			
			double[] gradient = new double[ numTopics ];
			double maxDerivative = Double.NEGATIVE_INFINITY;
			for (int row = 0; row < numTopics; row++) {
				gradient[row] = wordTimesAnchors[row];
				for (int col = 0; col < numTopics; col++) {
					gradient[row] -= conditionalProb[col] * anchorsTimesAnchors[row][col];
				}
				sumSquaredError += gradient[row] * gradient[row];

				//System.out.format("%f ", gradient[row]);
				gradient[row] *= 2.0 * learningRate;

				if (gradient[row] > maxDerivative) { maxDerivative = gradient[row]; }
			}
			//			System.out.println();

			double sum = 0.0;
			for (int row = 0; row < numTopics; row++) {
				conditionalProb[row] = Math.log(conditionalProb[row]) + gradient[row];
				sum += Math.exp(conditionalProb[row]);
			}

			double normalizer = Math.log(sum);
			for (int row = 0; row < numTopics; row++) {
				conditionalProb[row] = Math.exp(conditionalProb[row] - normalizer);
			}

			iteration++;
		}
		double entropy = 0.0;
		for (int topic = 0; topic < numTopics; topic++) {
			if (conditionalProb[topic] > 0.0) {
				entropy -= conditionalProb[topic] * Math.log(conditionalProb[topic]);
			}
		}

		System.err.format("%d\t%d\t%f\t%f\n", word, iteration, Math.log(sumSquaredError), entropy);

		return conditionalProb;
	}

	public static BigramProbabilityMatrix getMatrix(InstanceList instances, int numProjections, double sparsity, Randoms random) {

		BigramProbabilityMatrix matrix;

		if (numProjections == 0) {
			matrix = new BigramProbabilityMatrix(instances.getDataAlphabet());
		}
		else {
			if (sparsity == 1.0) {
				matrix = new GaussianRandomProjection(instances.getDataAlphabet(), numProjections, random);
			}
			else {
				// For this projection, sparsity is the number of non-zeros
				int nonZeros = (int) Math.floor(sparsity * numProjections);
				matrix = new FixedSparseRandomProjection(instances.getDataAlphabet(), numProjections, nonZeros, random);
			}
		}

		matrix.load(instances);
		//System.out.println("built matrix");
		matrix.rowNormalize();
		//System.out.println("normalized");

		return matrix;
	}


	public static void main (String[] args) throws Exception {
		CommandOption.setSummary (SpectralLDA.class, "Find anchor words and topic-word distributions associated with them");
		CommandOption.process (SpectralLDA.class, args);

		long startTime = System.currentTimeMillis();
		InstanceList instances = InstanceList.load(new File(inputFile.value));
		//System.out.println(instances.getDataAlphabet().size());

		Randoms random;
		if (randomSeedOption.value == 0) {
			random = new Randoms();
		}
		else {
			random = new Randoms(randomSeedOption.value);
		}

		BigramProbabilityMatrix matrix = getMatrix(instances, randomProjections.value, randomProjectionSparsity.value, random);

		StabilizedGS orthogonalizer = new StabilizedGS(matrix, minimumDocumentFrequency.value);
		System.out.format("%d / %d words above document cutoff for anchor words\n", orthogonalizer.getNumInterestingRows(), matrix.numWords);
		System.out.println("Finding anchor words");

		orthogonalizer.orthogonalize(numTopicsOption.value);

		if (anchorWordsFile.value != null) {
			orthogonalizer.writeAnchors(new File(anchorWordsFile.value));
		}

		// The orthogonalization modified the matrix in place, so reload it.
		orthogonalizer.clearMatrix();  // save some memory
		matrix = getMatrix(instances, randomProjections.value, randomProjectionSparsity.value, random);
		SpectralLDA recover = new SpectralLDA(matrix, orthogonalizer);

		/*
		PrintWriter matrixWriter = new PrintWriter("matrix.txt");
		for (int topic = 0; topic < recover.numTopics; topic++) {
			Formatter out = new Formatter();
			for (int i = 0; i < matrix.numColumns; i++) {
				out.format("%e ", matrix.weights[ orthogonalizer.basisVectorIndices[topic] ][i]);
			}
			matrixWriter.println(out);
		}
		matrixWriter.close();
		*/

		/*
		matrixWriter = new PrintWriter("non-anchors.txt");
		for (int word = 0; word < 10; word++) {
			Formatter out = new Formatter();
			for (int i = 0; i < matrix.numColumns; i++) {
				out.format("%e ", matrix.weights[ word ][i]);
			}
			matrixWriter.println(out);
		}
		matrixWriter.close();		
		*/

		PrintWriter out = new PrintWriter(new File(topicsFile.value));

		System.out.println("Finding topics");

		IDSorter[][] topicSortedWords = new IDSorter[recover.numTopics][matrix.numWords];

		for (int word = 0; word < matrix.numWords; word++) {
			double wordProb = matrix.unigramProbability(word);
			double[] weights = recover.recover(word);

			for (int topic = 0; topic < weights.length; topic++) {
				//out.format("%d:%.8f ", topic, wordProb * weights[topic]);
				out.print(wordProb * weights[topic] + "\t");
				topicSortedWords[topic][word] = new IDSorter(word, wordProb * weights[topic]);
			}
			out.println();

			if (word > 0 && word % 1000 == 0) {
				System.out.format("[%d] %s\n", word, instances.getDataAlphabet().lookupObject(word));
			}
		}
		out.close();

		Alphabet alphabet = instances.getDataAlphabet();

		for (int topic = 0; topic < recover.numTopics; topic++) {
			Arrays.sort(topicSortedWords[topic]);
			StringBuilder builder = new StringBuilder();
			for (int i = 0; i < 20; i++) {
				builder.append(alphabet.lookupObject(topicSortedWords[topic][i].getID()) + " ");
			}
			System.out.format("%d\t%s\t%s\n", topic, alphabet.lookupObject(orthogonalizer.basisVectorIndices[topic]), builder);
		}

		System.out.format("%d ms\n", System.currentTimeMillis() - startTime);
	}
	
}