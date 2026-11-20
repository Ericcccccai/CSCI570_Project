import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * CSCI 570 Fall 2026 Final Project - Sequence Alignment, memory-efficient version.
 *
 * Divide and conquer + dynamic programming (as in the lecture slides):
 *   1. Split X in half: X_L and X_R.
 *   2. With a DP that keeps only two rows, compute
 *        prefix[k] = OPT(X_L, y1..yk)        for every k = 0..n   (forward pass)
 *        suffix[k] = OPT(X_R, y(k+1)..yn)    for every k = 0..n   (backward pass)
 *   3. The optimal split point of Y is the k that minimizes prefix[k] + suffix[k].
 *   4. Recursively align (X_L, y1..yk) and (X_R, y(k+1)..yn) and concatenate.
 * Small subproblems are solved directly with the basic DP (see THRESHOLD).
 * Time O(mn), memory O(m + n).
 *
 * Usage: java Efficient <input file> <output file>
 */
public class Efficient {

    // Gap penalty and mismatch costs are fixed by the project description.
    private static final int DELTA = 30;
    private static final int[][] ALPHA = {
        //  A    C    G    T
        {   0, 110,  48,  94 },  // A
        { 110,   0, 118,  48 },  // C
        {  48, 118,   0, 110 },  // G
        {  94,  48, 110,   0 }   // T
    };

    /**
     * Base-case threshold. A subproblem of size m' + n' <= THRESHOLD (or one in
     * which a string has length <= 1) is solved directly with the basic DP
     * instead of being split further. See Summary.pdf for how it was chosen.
     */
    private static final int THRESHOLD = 64;

    // The two input strings, as characters and as codes 0..3 (A, C, G, T).
    private static char[] xChars;
    private static char[] yChars;
    private static byte[] xCode;
    private static byte[] yCode;

    // Reusable O(n) rows for the forward and backward passes.
    private static int[] forwardA;
    private static int[] forwardB;
    private static int[] backwardA;
    private static int[] backwardB;

    // Reusable buffers for the base-case basic DP and its backtracking.
    private static int[] baseTable;
    private static char[] traceFirst;
    private static char[] traceSecond;

    // The alignment is built left to right while the recursion runs.
    private static StringBuilder outFirst;
    private static StringBuilder outSecond;

    /** Holds the answer: the minimum cost and the two aligned strings. */
    private static final class Alignment {
        final int cost;
        final String first;
        final String second;

        Alignment(int cost, String first, String second) {
            this.cost = cost;
            this.first = first;
            this.second = second;
        }
    }

    public static void main(String[] args) throws IOException {
        String[] strings = generateInputStrings(args[0]);

        // Clean up garbage left over from reading/generating the input, so the
        // memory reading below reflects the alignment algorithm itself.
        System.gc();

        double beforeUsedMem = getMemoryInKB();
        double startTime = getTimeInMilliseconds();

        Alignment alignment = efficientSolution(strings[0], strings[1]);

        double afterUsedMem = getMemoryInKB();
        double endTime = getTimeInMilliseconds();

        double totalUsage = afterUsedMem - beforeUsedMem;
        double totalTime = endTime - startTime;

        writeOutput(args[1], alignment, totalTime, totalUsage);
    }

    // ------------------------------------------------------------------
    // Input string generator (Part I-B of the project description)
    // ------------------------------------------------------------------

    /**
     * Reads the input file and builds the two strings. Each number n on the
     * lines after a base string means: insert a copy of the current string into
     * itself right after index n (0-indexed), which doubles its length.
     */
    static String[] generateInputStrings(String inputPath) throws IOException {
        List<String> lines = new ArrayList<>();
        for (String raw : Files.readAllLines(Paths.get(inputPath), StandardCharsets.ISO_8859_1)) {
            // Keep only letters and digits; this also drops \r from Windows line endings.
            String line = raw.replaceAll("[^A-Za-z0-9]", "").toUpperCase();
            if (!line.isEmpty()) {
                lines.add(line);
            }
        }

        String[] result = new String[2];
        int pos = 0;
        for (int s = 0; s < 2; s++) {
            StringBuilder current = new StringBuilder(lines.get(pos++));
            while (pos < lines.size() && Character.isDigit(lines.get(pos).charAt(0))) {
                int index = Integer.parseInt(lines.get(pos++));
                current.insert(index + 1, current.toString());
            }
            result[s] = current.toString();
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Memory-efficient solution
    // ------------------------------------------------------------------

    static Alignment efficientSolution(String x, String y) {
        int m = x.length();
        int n = y.length();
        xChars = x.toCharArray();
        yChars = y.toCharArray();
        xCode = encode(xChars);
        yCode = encode(yChars);

        forwardA = new int[n + 1];
        forwardB = new int[n + 1];
        backwardA = new int[n + 1];
        backwardB = new int[n + 1];
        traceFirst = new char[m + n];
        traceSecond = new char[m + n];

        // One table, allocated once, big enough for every base case:
        // m' + n' <= THRESHOLD needs at most (ceil(THRESHOLD/2) + 1)^2 cells, and a
        // subproblem with a string of length 1 needs at most 2 * (max(m, n) + 1).
        long half = (THRESHOLD + 1) / 2 + 1;
        long cells = Math.max(half * half, 2L * (Math.max(m, n) + 1));
        cells = Math.min(cells, (long) (m + 1) * (n + 1));
        baseTable = new int[(int) cells];
        outFirst = new StringBuilder(m + n);
        outSecond = new StringBuilder(m + n);

        divideAndConquer(0, m, 0, n);

        String first = outFirst.toString();
        String second = outSecond.toString();
        return new Alignment(alignmentCost(first, second), first, second);
    }

    /** Appends an optimal alignment of x[xLo..xHi) and y[yLo..yHi) to the output. */
    private static void divideAndConquer(int xLo, int xHi, int yLo, int yHi) {
        int m = xHi - xLo;
        int n = yHi - yLo;

        if (m == 0) {                       // only Y left: every y character faces a gap
            for (int j = yLo; j < yHi; j++) {
                outFirst.append('_');
                outSecond.append(yChars[j]);
            }
            return;
        }
        if (n == 0) {                       // only X left: every x character faces a gap
            for (int i = xLo; i < xHi; i++) {
                outFirst.append(xChars[i]);
                outSecond.append('_');
            }
            return;
        }
        if (m + n <= THRESHOLD || m == 1 || n == 1) {
            basicAlign(xLo, xHi, yLo, yHi);  // small enough: solve directly
            return;
        }

        // Divide: split X in half and find the best place to split Y.
        int xMid = xLo + m / 2;
        int[] prefix = forwardCosts(xLo, xMid, yLo, yHi);
        int[] suffix = backwardCosts(xMid, xHi, yLo, yHi);

        int bestK = 0;
        int bestCost = Integer.MAX_VALUE;
        for (int k = 0; k <= n; k++) {
            int total = prefix[k] + suffix[k];
            if (total < bestCost) {
                bestCost = total;
                bestK = k;
            }
        }

        // Conquer: the two halves are independent; the left one is appended first.
        int yMid = yLo + bestK;
        divideAndConquer(xLo, xMid, yLo, yMid);
        divideAndConquer(xMid, xHi, yMid, yHi);
    }

    /**
     * Forward pass with two rows. Returns row r where
     * r[k] = OPT(x[xLo..xHi), y[yLo..yLo+k)) for k = 0..n.
     */
    private static int[] forwardCosts(int xLo, int xHi, int yLo, int yHi) {
        int n = yHi - yLo;
        int[] prev = forwardA;
        int[] cur = forwardB;
        for (int k = 0; k <= n; k++) {
            prev[k] = k * DELTA;
        }
        for (int i = xLo; i < xHi; i++) {
            int[] mismatch = ALPHA[xCode[i]];
            cur[0] = prev[0] + DELTA;
            for (int k = 1; k <= n; k++) {
                int best = prev[k - 1] + mismatch[yCode[yLo + k - 1]];
                int xGap = prev[k] + DELTA;
                if (xGap < best) {
                    best = xGap;
                }
                int yGap = cur[k - 1] + DELTA;
                if (yGap < best) {
                    best = yGap;
                }
                cur[k] = best;
            }
            int[] swap = prev;
            prev = cur;
            cur = swap;
        }
        return prev;
    }

    /**
     * Backward pass with two rows, running from the ends of both strings.
     * Returns row r where r[k] = OPT(x[xLo..xHi), y[yLo+k..yHi)) for k = 0..n,
     * so it is indexed the same way as the forward row.
     */
    private static int[] backwardCosts(int xLo, int xHi, int yLo, int yHi) {
        int n = yHi - yLo;
        int[] prev = backwardA;
        int[] cur = backwardB;
        for (int k = 0; k <= n; k++) {
            prev[k] = (n - k) * DELTA;
        }
        for (int i = xHi - 1; i >= xLo; i--) {
            int[] mismatch = ALPHA[xCode[i]];
            cur[n] = prev[n] + DELTA;
            for (int k = n - 1; k >= 0; k--) {
                int best = prev[k + 1] + mismatch[yCode[yLo + k]];
                int xGap = prev[k] + DELTA;
                if (xGap < best) {
                    best = xGap;
                }
                int yGap = cur[k + 1] + DELTA;
                if (yGap < best) {
                    best = yGap;
                }
                cur[k] = best;
            }
            int[] swap = prev;
            prev = cur;
            cur = swap;
        }
        return prev;
    }

    /** Base case: basic DP (full table) on x[xLo..xHi) and y[yLo..yHi), then backtrack. */
    private static void basicAlign(int xLo, int xHi, int yLo, int yHi) {
        int m = xHi - xLo;
        int n = yHi - yLo;
        int width = n + 1;
        int[] opt = baseTable;  // preallocated in efficientSolution, always large enough

        for (int k = 0; k <= n; k++) {
            opt[k] = k * DELTA;
        }
        for (int i = 1; i <= m; i++) {
            int row = i * width;
            int prevRow = row - width;
            opt[row] = i * DELTA;
            int[] mismatch = ALPHA[xCode[xLo + i - 1]];
            for (int k = 1; k <= n; k++) {
                int best = opt[prevRow + k - 1] + mismatch[yCode[yLo + k - 1]];
                int xGap = opt[prevRow + k] + DELTA;
                if (xGap < best) {
                    best = xGap;
                }
                int yGap = opt[row + k - 1] + DELTA;
                if (yGap < best) {
                    best = yGap;
                }
                opt[row + k] = best;
            }
        }

        // Backtrack from the bottom-right corner (this produces the alignment reversed).
        int len = 0;
        int i = m;
        int k = n;
        while (i > 0 || k > 0) {
            int here = opt[i * width + k];
            if (i > 0 && k > 0
                    && here == opt[(i - 1) * width + (k - 1)] + ALPHA[xCode[xLo + i - 1]][yCode[yLo + k - 1]]) {
                traceFirst[len] = xChars[xLo + i - 1];
                traceSecond[len] = yChars[yLo + k - 1];
                i--;
                k--;
            } else if (i > 0 && here == opt[(i - 1) * width + k] + DELTA) {
                traceFirst[len] = xChars[xLo + i - 1];
                traceSecond[len] = '_';
                i--;
            } else {
                traceFirst[len] = '_';
                traceSecond[len] = yChars[yLo + k - 1];
                k--;
            }
            len++;
        }
        for (int p = len - 1; p >= 0; p--) {
            outFirst.append(traceFirst[p]);
            outSecond.append(traceSecond[p]);
        }
    }

    /** Cost of a finished alignment: DELTA per gap, ALPHA per matched pair. */
    private static int alignmentCost(String first, String second) {
        int cost = 0;
        for (int p = 0; p < first.length(); p++) {
            char a = first.charAt(p);
            char b = second.charAt(p);
            if (a == '_' || b == '_') {
                cost += DELTA;
            } else {
                cost += ALPHA[code(a)][code(b)];
            }
        }
        return cost;
    }

    /** Maps A, C, G, T to 0, 1, 2, 3 (the row/column order of ALPHA). */
    private static int code(char c) {
        switch (c) {
            case 'A': return 0;
            case 'C': return 1;
            case 'G': return 2;
            default:  return 3;  // 'T'
        }
    }

    private static byte[] encode(char[] s) {
        byte[] codes = new byte[s.length];
        for (int k = 0; k < s.length; k++) {
            codes[k] = (byte) code(s[k]);
        }
        return codes;
    }

    // ------------------------------------------------------------------
    // Output, memory and time (based on the sample code in the description)
    // ------------------------------------------------------------------

    private static void writeOutput(String outputPath, Alignment alignment,
                                    double timeMs, double memoryKB) throws IOException {
        File parent = new File(outputPath).getAbsoluteFile().getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputPath), StandardCharsets.US_ASCII)) {
            writer.write(Integer.toString(alignment.cost));
            writer.newLine();
            writer.write(alignment.first);
            writer.newLine();
            writer.write(alignment.second);
            writer.newLine();
            writer.write(Double.toString(timeMs));
            writer.newLine();
            writer.write(Double.toString(memoryKB));
            writer.newLine();
        }
    }

    /** Heap memory currently in use, in kilobytes (1 KB = 1024 bytes). */
    private static double getMemoryInKB() {
        double total = Runtime.getRuntime().totalMemory();
        return (total - Runtime.getRuntime().freeMemory()) / 1024.0;
    }

    /** Current time in milliseconds (System.nanoTime() is in nanoseconds). */
    private static double getTimeInMilliseconds() {
        return System.nanoTime() / 1e6;
    }
}
