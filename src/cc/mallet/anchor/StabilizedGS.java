package cc.mallet.anchor;

import java.io.*;
import cc.mallet.util.*;
import cc.mallet.types.*;

public class StabilizedGS {

	BigramProbabilityMatrix matrix;
	double[] rowSquaredSums;
	int[] basisVectorIndices;
	boolean[] rowIsBasisVector;
	boolean[] rowIsInteresting;
	int numInterestingRows;
	int minDocs;

	public StabilizedGS(BigramProbabilityMatrix matrix, int minDocs) {
		this.matrix = matrix;
		this.minDocs = minDocs;
		rowSquaredSums = new double[matrix.numWords];
		rowIsBasisVector = new boolean[matrix.numWords];
		rowIsInteresting = new boolean[matrix.numWords];

		numInterestingRows = 0;
		for (int row = 0; row < matrix.numWords; row++) {
			if (matrix.documentFrequencies[row] >= minDocs) {
				rowIsInteresting[row] = true;
				numInterestingRows++;
				for (int col = 0; col < matrix.numColumns; col++) {
					rowSquaredSums[row] += matrix.weights[row][col] * matrix.weights[row][col];
				}
			}
		}
	}

	public int[] getBasisVectorIndices() { return basisVectorIndices; }
	public boolean[] getRowIsBasisVector() { return rowIsBasisVector; }
	public int getNumInterestingRows() { return numInterestingRows; }

	/** Remove link to matrix for garbage collection.
		Use only if memory is limited, as this could lead to null pointer exceptions */
	public void clearMatrix() { matrix = null; }

	/** Return an array containing the indices of the basis vectors */
	public int[] orthogonalize(int numBasisVectors) {
		
		basisVectorIndices = new int[numBasisVectors];

		for (int i = 0; i < numBasisVectors; i++) {
			
			// Find max
			double maxValue = 0.0;
			int maxRow = 0;
			for (int row = 0; row < matrix.numWords; row++) {
				if (! rowIsBasisVector[row] && rowIsInteresting[row] && rowSquaredSums[row] > maxValue) {
					maxRow = row;
					maxValue = rowSquaredSums[row];
				}
			}

			// maxRow will be our next basis vector
			basisVectorIndices[i] = maxRow;
			rowIsBasisVector[maxRow] = true;

			System.out.format("%d\t%s\n", maxRow, matrix.vocabulary.lookupObject(maxRow));

			// Normalize the vector
			double normalizer = 1.0 / Math.sqrt(rowSquaredSums[maxRow]);
			for (int col = 0; col < matrix.numColumns; col++) {
				matrix.weights[maxRow][col] *= normalizer;
			}

			for (int row = 0; row < matrix.numWords; row++) {
				if (! rowIsBasisVector[row] && rowIsInteresting[row]) {
					double newSumSquares = 0.0;
					
					// Get the inner product
					double innerProduct = 0.0;
					for (int col = 0; col < matrix.numColumns; col++) {
						innerProduct += matrix.weights[maxRow][col] * matrix.weights[row][col];
					}

					rowSquaredSums[row] = 0.0;
					// Now loop through again to subtract the projection
					for (int col = 0; col < matrix.numColumns; col++) {
						matrix.weights[row][col] -= innerProduct * matrix.weights[maxRow][col];
						rowSquaredSums[row] += matrix.weights[row][col] * matrix.weights[row][col];
					}
				}
			}
			
		}
		
		return basisVectorIndices;
	}

	public void writeAnchors(File anchorsFile) throws IOException {
		PrintWriter out = new PrintWriter(anchorsFile);
		for (int index: basisVectorIndices) {
			out.format("%d\t%s\n", index, matrix.vocabulary.lookupObject(index));
		}
		out.close();
	}

}