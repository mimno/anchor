package cc.mallet.anchor;

import cc.mallet.util.Randoms;
import cc.mallet.types.*;
import gnu.trove.map.hash.*;
import java.util.*;
import java.io.*;

public class GaussianRandomProjection extends BigramProbabilityMatrix {

	double[][] projectionMatrix;
	double[] rowSums;
	
	public GaussianRandomProjection (Alphabet a, int randomProjections, Randoms random) {
		vocabulary = a;
		numWords = vocabulary.size();
		numColumns = randomProjections;
		weights = new double[numWords][numColumns];
		
		wordCounts = new int[numWords];
		documentFrequencies = new int[numWords];
		rowSums = new double[numWords];

		projectionMatrix = new double[numWords][randomProjections];

		for (int word = 0; word < numWords; word++) {
			for (int col = 0; col < randomProjections; col++) {
				projectionMatrix[word][col] = random.nextGaussian();
			}
		}
	}

	public void load(InstanceList instances) {
		int numDocuments = instances.size();

		long startTime = System.currentTimeMillis();
		for (Instance instance: instances) {
			//HashMap<Integer,Integer> typeCounts = new HashMap<Integer,Integer>();

			TIntIntHashMap typeCounts = new TIntIntHashMap();
			FeatureSequence tokens = (FeatureSequence) instance.getData();
			int length = tokens.getLength();

			// Skip single-word documents
			if (length < 10) { continue; }

			totalTokens += length;

			for (int position = 0; position < length; position++) {
				int type = tokens.getIndexAtPosition(position);
				wordCounts[type]++;
				typeCounts.adjustOrPutValue(type, 1, 1);
			}

			double coefficient = 2.0 / (length * (length - 1));

			int[] types = typeCounts.keys();
			int[] counts = typeCounts.values(); 

			// Multiply the counts by the random projection matrix
			double[] projectedSignature = new double[numColumns];
			for (int i = 0; i < types.length; i++) {
				int type = types[i];
				documentFrequencies[type]++;
				for (int j = 0; j < numColumns; j++) {
					projectedSignature[j] += counts[i] * projectionMatrix[type][j];
				}
				rowSums[type] += coefficient * counts[i] * (length - counts[i]);
			}
			
			// Rescale for the sparse random projection and the length of the document
			for (int j = 0; j < numColumns; j++) {
				projectedSignature[j] *= coefficient; 
			}

			// Now multiply by the counts again, subtracting the term for the word itself
			for (int i = 0; i < types.length; i++) {
				int type = types[i];
				for (int j = 0; j < numColumns; j++) {
					weights[type][j] += counts[i] * (projectedSignature[j] - coefficient * counts[i] * projectionMatrix[type][j]);
				}
			}
			/*
			  for (int firstType: typeCounts.keySet()) {
			  for (int secondType: typeCounts.keySet()) {
			  if (firstType != secondType) {
			  weights[firstType][secondType] += coefficient * typeCounts.get(firstType) * typeCounts.get(secondType);
			  }
			  }
			  }
			*/

		}
		//System.err.format("%d ms\n", (System.currentTimeMillis() - startTime));

	}

	public double unigramProbability(int type) {
		return ((double) wordCounts[type]) / totalTokens;
	}

	/** We need to keep track of what the magnitude of the original row was, not the 
		magnitude of the projected row */
	public void rowNormalize() {
		for (int row = 0; row < numWords; row++) {
			double normalizer = 1.0 / rowSums[row];
			for (int col = 0; col < numColumns; col++) {
                weights[row][col] *= normalizer;
            }
        }
	}

	public void naiveRowNormalize() {
		for (int row = 0; row < numWords; row++) {
			double sum = 0.0;
            for (int col = 0; col < numColumns; col++) {
				sum += Math.abs(weights[row][col]);
			}
			double normalizer = 1.0 / sum;
            for (int col = 0; col < numColumns; col++) {
				weights[row][col] *= normalizer;
			}
		}
	}

	public static void main (String[] args) throws Exception {
		//Alphabet vocabulary = AlphabetFactory.loadFromFile(new File(args[0]));
		InstanceList instances = InstanceList.load(new File(args[0]));

		GaussianRandomProjection matrix = new GaussianRandomProjection(instances.getDataAlphabet(), 1000, new Randoms());

		matrix.load(instances);

		/*
		  for (int row = 0; row < matrix.numWords; row++) {
		  Formatter out = new Formatter();
		  for (int col = 0; col < matrix.numWords; col++) {
		  out.format("%.9f ", matrix.weights[row][col]);
		  }
		  System.out.println(out);
		  }
		*/

	}

}
