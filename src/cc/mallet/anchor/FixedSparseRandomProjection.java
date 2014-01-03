package cc.mallet.anchor;

import cc.mallet.util.Randoms;
import cc.mallet.types.*;
import gnu.trove.map.hash.*;
import java.util.*;
import java.io.*;

public class FixedSparseRandomProjection extends BigramProbabilityMatrix {

	int[][] projectionMatrix;
	double squareRootSparsity;

	int nonZeros;
	
	double[] rowSums;
	
	public FixedSparseRandomProjection (Alphabet a, int randomProjections, int nonZeros, Randoms random) {
		System.err.println("Calculating a sparse random projection");

		vocabulary = a;
		numWords = vocabulary.size();
		numColumns = randomProjections;
		this.nonZeros = nonZeros;
		weights = new double[numWords][numColumns];
		
		wordCounts = new int[numWords];
		documentFrequencies = new int[numWords];
		rowSums = new double[numWords];

		projectionMatrix = new int[numWords][nonZeros];

		//squareRootSparsity = Math.sqrt((double) nonZeros / randomProjections);
		squareRootSparsity = Math.sqrt((double) randomProjections / nonZeros);

		// Make an array with the numbers 0 .. (dictionary size - 1)
		int[] allProjectionIndices = new int[randomProjections];
		for (int i = 0; i < randomProjections; i++) {
			allProjectionIndices[i] = i;
		}

		for (int word = 0; word < numWords; word++) {
			// For each word we're going to randomly shuffle the projection indices array up to the first [nonZeros] entries.
			for (int col = 0; col < nonZeros; col++) {
				// Randomly swap the current index with one further down the array
				int swapCol = col + random.nextInt(randomProjections - col);
				int temp = allProjectionIndices[swapCol];
				allProjectionIndices[swapCol] = allProjectionIndices[col];
				allProjectionIndices[col] = temp;

				// And use the swapped index for the projection index.
				projectionMatrix[word][col] = allProjectionIndices[col];
				// Use a second range of values for indices that should be negated.
				if (random.nextUniform() > 0.5) {
					projectionMatrix[word][col] += randomProjections;
				}
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

			// Skip short documents
			if (length < 10) { continue; }

			totalTokens += length;

			for (int position = 0; position < length; position++) {
				int type = tokens.getIndexAtPosition(position);
				wordCounts[type]++;
				typeCounts.adjustOrPutValue(type, 1, 1);
			}

			double inverseNumPairs = 2.0 / (length * (length - 1));
			double coefficient = squareRootSparsity * 2.0 / (length * (length - 1));

			int[] types = typeCounts.keys();
			int[] counts = typeCounts.values(); 

			// Multiply the counts by the random projection matrix
			double[] projectedSignature = new double[numColumns];
			for (int i = 0; i < types.length; i++) {
				int type = types[i];
				documentFrequencies[type]++;

				for (int j = 0; j < nonZeros; j++) {
					int projectionIndex = projectionMatrix[type][j];
					if (projectionIndex >= numColumns) {
						// If it's in this range, it should be negated
						projectedSignature[projectionIndex - numColumns] += -counts[i];
					}
					else {
						projectedSignature[projectionIndex] += counts[i];
					}
				}
				rowSums[type] += inverseNumPairs * counts[i] * (length - counts[i]);
			}
			
			// Rescale for the sparse random projection and the length of the document
			for (int j = 0; j < numColumns; j++) {
				projectedSignature[j] *= coefficient; 
			}

			// Now multiply by the counts again, subtracting the term for the word itself
			for (int i = 0; i < types.length; i++) {
				int type = types[i];
				for (int denseIndex = 0; denseIndex < numColumns; denseIndex++) {
					weights[type][denseIndex] += counts[i] * projectedSignature[denseIndex];
				}
				// Remove the diagonal elements as needed
				for (int sparseIndex = 0; sparseIndex < nonZeros; sparseIndex++) {
					int denseIndex = projectionMatrix[type][sparseIndex];
					if (denseIndex >= numColumns) {
						weights[type][denseIndex - numColumns] -= -coefficient * counts[i] * counts[i];
					}
					else {
						weights[type][denseIndex] -= coefficient * counts[i] * counts[i];
					}
				}
			}

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
			double normalizer = 1.0 / (rowSums[row] * Math.sqrt(numColumns));
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

		FixedSparseRandomProjection matrix = new FixedSparseRandomProjection(instances.getDataAlphabet(), 1000, 10, new Randoms());

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
