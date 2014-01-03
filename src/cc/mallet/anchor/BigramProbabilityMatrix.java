package cc.mallet.anchor;

import cc.mallet.types.*;
import gnu.trove.map.hash.*;
import java.util.*;
import java.io.*;

public class BigramProbabilityMatrix {
	Alphabet vocabulary;
	
	int numWords;
	int numColumns;
	double[][] weights;

	int totalTokens;
	int[] wordCounts;
	int[] documentFrequencies;

	public BigramProbabilityMatrix() { }

	public BigramProbabilityMatrix(Alphabet a) {
		vocabulary = a;
		
		numWords = vocabulary.size();
		numColumns = numWords;
		weights = new double[numWords][numWords];

		wordCounts = new int[numWords];
		documentFrequencies = new int[numWords];
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
				/*
				if (! typeCounts.containsKey(type)) {
					typeCounts.put(type, 1);
				}
				else {
					typeCounts.put(type, typeCounts.get(type) + 1);
				}
				*/
			}

			double coefficient = 2.0 / (length * (length - 1));

			int[] types = typeCounts.keys();
			int[] counts = typeCounts.values(); 

			/*
			if (typeCounts.get(1770) > 0) {
				System.out.format("1770: %d 3667: %d [%d]\n", typeCounts.get(1770), typeCounts.get(3667), length);
			}
			*/
			
			for (int i = 0; i < types.length; i++) {
				int firstType = types[i];
				documentFrequencies[firstType]++;

				for (int j = i+1; j < types.length; j++) {
					int secondType = types[j];
					double value = coefficient * counts[i] * counts[j];

					if (value < 0.0) {
						System.out.format("(2.0 / (%d * (%d - 1))) * %d * %d\n", length, length, counts[i], counts[j]);
					}

					weights[firstType][secondType] += value;
					weights[secondType][firstType] += value;
					
					/*
					if ((firstType == 1770 && secondType == 3667) || (firstType == 3667 && secondType == 1770)) {
						for (int position = 0; position < length; position++) {
							System.out.format("%s ", vocabulary.lookupObject(tokens.getIndexAtPosition(position)));
						}

						System.out.println(value);
					}
					*/
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

	public void rowNormalize() {
		for (int row = 0; row < numWords; row++) {
			double sum = 0.0;
            for (int col = 0; col < numWords; col++) {
				sum += weights[row][col];
			}
			double normalizer = 1.0 / sum;
            for (int col = 0; col < numWords; col++) {
				weights[row][col] *= normalizer;
			}
			
		}
	}

	public double euclideanDistance(int firstRow, int secondRow) {
		double sum = 0.0;
		for (int col = 0; col < numColumns; col++) {
			double diff = weights[firstRow][col] - weights[secondRow][col];
			sum += diff * diff;
		}
		return Math.sqrt(sum);
	}

	public String topWordsFor(int row, int n) {
		Formatter formatter = new Formatter();

		IDSorter[] sortedWords = new IDSorter[numWords];
		for (int col = 0; col < numWords; col++) {
			sortedWords[col] = new IDSorter(col, weights[row][col]);
		}
		Arrays.sort(sortedWords);
		
		if (n > numWords) { n = numWords; }
		for (int i = 0; i < n; i++) {
			formatter.format("%s %d (%f) ", vocabulary.lookupObject(sortedWords[i].getID()), sortedWords[i].getID(), sortedWords[i].getWeight());
		}
		return formatter.toString();
	}

	public Alphabet getVocabulary() { return vocabulary; }

	public static void main (String[] args) throws Exception {
		//Alphabet vocabulary = AlphabetFactory.loadFromFile(new File(args[0]));
		InstanceList instances = InstanceList.load(new File(args[0]));

		BigramProbabilityMatrix matrix = new BigramProbabilityMatrix(instances.getDataAlphabet());

		matrix.load(instances);

		System.out.println(matrix.topWordsFor(1770, 20));

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