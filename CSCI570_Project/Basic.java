import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * CSCI 570 Fall 2026 Final Project - Sequence Alignment, basic version.
 *
 * Bottom-up dynamic programming over the full (m+1) x (n+1) table, then a
 * top-down pass (backtracking) from OPT(m, n) to recover one optimal alignment.
 * Time O(mn), memory O(mn).
 *
 * Usage: java Basic <input file> <output file>
 */
public class Basic {

    // Gap penalty and mismatch costs are fixed by the project description.
    private static final int DELTA = 30;
    private static final int[][] ALPHA = {
        //  A    C    G    T
        {   0, 110,  48,  94 },  // A
        { 110,   0, 118,  48 },  // C
        {  48, 118,   0, 110 },  // G
        {  94,  48, 110,   0 }   // T
    };

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

        Alignment alignment = basicSolution(strings[0], strings[1]);

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
    // Basic dynamic programming solution
    // ------------------------------------------------------------------

    /**
     * OPT(i, j) = minimum cost of aligning x1..xi with y1..yj
     *           = min( ALPHA[xi][yj] + OPT(i-1, j-1),   // match xi with yj
     *                  DELTA + OPT(i-1, j),             // xi is a gap
     *                  DELTA + OPT(i, j-1) )            // yj is a gap
     * with OPT(i, 0) = i * DELTA and OPT(0, j) = j * DELTA.
     */
    static Alignment basicSolution(String x, String y) {
        int m = x.length();
        int n = y.length();
        int[] xc = encode(x);
        int[] yc = encode(y);

        // Bottom-up pass: fill the whole table.
        int[][] opt = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            opt[i][0] = i * DELTA;
        }
        for (int j = 0; j <= n; j++) {
            opt[0][j] = j * DELTA;
        }
        for (int i = 1; i <= m; i++) {
            int[] prevRow = opt[i - 1];
            int[] row = opt[i];
            int[] mismatch = ALPHA[xc[i - 1]];
            for (int j = 1; j <= n; j++) {
                int best = prevRow[j - 1] + mismatch[yc[j - 1]];
                int xGap = prevRow[j] + DELTA;
                if (xGap < best) {
                    best = xGap;
                }
                int yGap = row[j - 1] + DELTA;
                if (yGap < best) {
                    best = yGap;
                }
                row[j] = best;
            }
        }

        // Top-down pass: walk back from OPT(m, n) to recover the alignment.
        StringBuilder first = new StringBuilder(m + n);
        StringBuilder second = new StringBuilder(m + n);
        int i = m;
        int j = n;
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && opt[i][j] == opt[i - 1][j - 1] + ALPHA[xc[i - 1]][yc[j - 1]]) {
                first.append(x.charAt(i - 1));
                second.append(y.charAt(j - 1));
                i--;
                j--;
            } else if (i > 0 && opt[i][j] == opt[i - 1][j] + DELTA) {
                first.append(x.charAt(i - 1));
                second.append('_');
                i--;
            } else {
                first.append('_');
                second.append(y.charAt(j - 1));
                j--;
            }
        }

        return new Alignment(opt[m][n], first.reverse().toString(), second.reverse().toString());
    }

    /** Maps A, C, G, T to 0, 1, 2, 3 (the row/column order of ALPHA). */
    private static int[] encode(String s) {
        int[] codes = new int[s.length()];
        for (int k = 0; k < s.length(); k++) {
            switch (s.charAt(k)) {
                case 'A': codes[k] = 0; break;
                case 'C': codes[k] = 1; break;
                case 'G': codes[k] = 2; break;
                default:  codes[k] = 3; break;  // 'T'
            }
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
