package cc.mallet.anchor;

import java.io.*;
import cc.mallet.util.*;
import cc.mallet.types.*;

public class StabilizedGS {

	static cc.mallet.util.CommandOption.String inputFile = new cc.mallet.util.CommandOption.String
		(StabilizedGS.class, "input", "FILENAME", true, null,
		 "The filename of a mallet instance list.", null);
	
	static cc.mallet.util.CommandOption.String outputFile = new cc.mallet.util.CommandOption.String
		(StabilizedGS.class, "anchor-indices", "FILENAME", true, "anchors.txt",
		 "The filename into which to write anchor words.", null);
	
	static cc.mallet.util.CommandOption.Integer numDimensionsOption = new cc.mallet.util.CommandOption.Integer
		(StabilizedGS.class, "num-dimensions", "INTEGER", true, 10,
		 "The number of dimensions (ie topics) to find.", null);
	
	static cc.mallet.util.CommandOption.Integer randomProjections = new cc.mallet.util.CommandOption.Integer
		(StabilizedGS.class, "num-random-projections", "INTEGER", true, 0,
		 "The dimensionality of a random projection. If 0, the original data will be used.", null);
	
	static cc.mallet.util.CommandOption.Double randomProjectionSparsity = new cc.mallet.util.CommandOption.Double
		(StabilizedGS.class, "random-projection-sparsity", "[0-1]", true, 0.1,
		 "The sparsity of the random projection: 0.1 means 10% of entries will be non-zero. If 0, a Gaussian RP will be used.", null);
	
	static cc.mallet.util.CommandOption.String randomProjectionType = new cc.mallet.util.CommandOption.String
		(StabilizedGS.class, "random-projection-type", "[gaussian|sparse|fixed]", true, "fixed",
		 "Which kind of random projection to use: 'guassian' is a dense matrix of standard normals, 'sparse' puts 1/0/-1 in each cell independently, 'fixed' puts a fixed number of non-zeros in each column.", null);

	static cc.mallet.util.CommandOption.Integer minimumDocumentFrequency = new cc.mallet.util.CommandOption.Integer
		(StabilizedGS.class, "min-docs", "INTEGER", true, 10,
		 "Don't consider words that appear in fewer than this many documents.", null);
	
	static cc.mallet.util.CommandOption.String matrixFilename = new cc.mallet.util.CommandOption.String
		(StabilizedGS.class, "matrix-file", "FILENAME", true, null,
		 "Write the data matrix as tab-delimited text to this file. If null, no file is written.", null);
	
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

			System.out.format("%d\t%f\t%s\n", maxRow, maxValue, matrix.vocabulary.lookupObject(maxRow));

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

	public static void main (String[] args) throws Exception {
		CommandOption.setSummary (StabilizedGS.class, "Do a Gram-Schmidt orthogonalization.");
        CommandOption.process (StabilizedGS.class, args);

		InstanceList instances = InstanceList.load(new File(inputFile.value));
		//System.out.println(instances.getDataAlphabet().size());

		BigramProbabilityMatrix matrix;

		if (randomProjections.value == 0) {
			matrix = new BigramProbabilityMatrix(instances.getDataAlphabet());
		}
		else {
			if (randomProjectionSparsity.value == 0 || "gaussian".startsWith(randomProjectionType.value)) {
				matrix = new GaussianRandomProjection(instances.getDataAlphabet(), randomProjections.value, new Randoms());
			}
			else if ("sparse".startsWith(randomProjectionType.value)) {
				// For this projection, the sparsity parameter is the inverse of the probability of a non-zero
				int sparsity = (int) Math.floor(1.0 / randomProjectionSparsity.value);
				matrix = new SparseRandomProjection(instances.getDataAlphabet(), randomProjections.value, sparsity, new Randoms());
			}
			else {
				// For this projection, sparsity is the number of non-zeros
				int sparsity = (int) Math.floor(randomProjectionSparsity.value * randomProjections.value);
				matrix = new FixedSparseRandomProjection(instances.getDataAlphabet(), randomProjections.value, sparsity, new Randoms());
			}
		}

		matrix.load(instances);
		//System.out.println("built matrix");
		matrix.rowNormalize();
		//System.out.println("normalized");
		
		if (matrixFilename.value != null) {
			PrintWriter out = new PrintWriter(new File(matrixFilename.value));

			for (int row = 0; row < matrix.numWords; row++) {
				StringBuilder builder = new StringBuilder();
				for (int col = 0; col < matrix.numColumns; col++) {
					builder.append(matrix.weights[row][col] + "\t");
				}
				out.println(builder);
			}

			out.close();
		}

		StabilizedGS orthogonalizer = new StabilizedGS(matrix, minimumDocumentFrequency.value);
		int[] indices = orthogonalizer.orthogonalize(numDimensionsOption.value);
		orthogonalizer.writeAnchors(new File(outputFile.value));
	}
	
}