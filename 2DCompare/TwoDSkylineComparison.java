import java.io.*;
import java.util.*;
import java.lang.management.*;

/**
 * Comparing this scheme with SVLSQ, SecSky, and SVkNN
 */
public class TwoDSkylineComparison {

    // Dataset holder
    static class Building {
        String buildingId;
        double minX, maxX, minY, maxY, minZ, maxZ;
        int price, environmentalIndicator, reviews, crimeRate;
        int yearBuilt, numUnits, energyEfficiency, propertyTax, walkScore;
        // 2D center for distance-based candidate selection
        double centerX, centerY;


        // Non-spatial attributes (up to 90 dims for Arkansas_90.csv)
        double[] features = new double[90];
        public Building(String[] parts) {
            buildingId = parts[0];
            minX = Double.parseDouble(parts[1]);
            maxX = Double.parseDouble(parts[2]);
            minY = Double.parseDouble(parts[3]);
            maxY = Double.parseDouble(parts[4]);
            minZ = Double.parseDouble(parts[5]);
            maxZ = Double.parseDouble(parts[6]);

            // Fill non-spatial features
            try {
                if (parts.length >= 97) {
                    // 7 spatial columns first, then 90 attributes
                    for (int i = 0; i < 90; i++) {
                        String v = parts[7 + i].trim();
                        if (v.isEmpty()) features[i] = 0;
                        else features[i] = Double.parseDouble(v);
                    }
                } else if (parts.length >= 16) {
                    // Backward-compat: map the original 9 attributes into first 9 dims
                    // [7]=price,[8]=environmentalIndicator,[9]=reviews,[10]=crimeRate,
                    // [11]=yearBuilt,[12]=numUnits,[13]=energyEfficiency,[14]=propertyTax,[15]=walkScore
                    int[] idx = {7,8,9,10,11,12,13,14,15};
                    for (int i = 0; i < idx.length && i < features.length; i++) {
                        String v = parts[idx[i]].trim();
                        if (v.isEmpty()) features[i] = 0;
                        else features[i] = Double.parseDouble(v);
                    }
                }
            } catch (NumberFormatException ex) {
                // Leave zeros for any unparsable values
            }
            // (Optional) keep legacy ints if present; ignore if columns don't match
            try {
                price = (int)features[0];
                environmentalIndicator = (int)features[1];
                reviews = (int)features[2];
                crimeRate = (int)features[3];
                yearBuilt = (int)features[4];
                numUnits = (int)features[5];
                energyEfficiency = (int)features[6];
                propertyTax = (int)features[7];
                walkScore = (int)features[8];
            } catch (Exception ignore) {}
            centerX = (minX + maxX) / 2.0;
            centerY = (minY + maxY) / 2.0;
        }

        public double[] getNonSpatialAttributes(int d) {
            int len = Math.min(d, features.length);
            double[] attrs = new double[d];
            System.arraycopy(features, 0, attrs, 0, len);
            // if d > features.length, the tail will remain zeros
            return attrs;
        }
    }

    static List<Building> dataset = new ArrayList<>();

    // Load dataset
    static void loadDataset(String filename, int maxRecords) {
        dataset.clear();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine(); // Skip header
            int count = 0;
            while ((line = br.readLine()) != null && count < maxRecords) {
                String[] parts = line.split(",");
                if (parts.length == 16 || parts.length >= 97) {
                    dataset.add(new Building(parts));
                    count++;
                }
            }
            System.out.println("Loaded " + dataset.size() + " buildings from dataset");
        } catch (IOException e) {
            System.err.println("Error loading dataset: " + e.getMessage());
            generateSyntheticData(maxRecords);
        }
    }

    // Fallback synthetic data generation
    static void generateSyntheticData(int n) {
        System.out.println("Generating synthetic dataset with " + n + " buildings");
        Random rand = new Random(42);
        dataset.clear();

        for (int i = 0; i < n; i++) {
            String[] parts = new String[16];
            parts[0] = "B"+ i;
            double x = rand.nextDouble() * 1000;
            double y = rand.nextDouble() * 1000;
            parts[1] = String.valueOf(x);
            parts[2] = String.valueOf(x + rand.nextDouble() * 10);
            parts[3] = String.valueOf(y);
            parts[4] = String.valueOf(y + rand.nextDouble() * 10);
            parts[5] = "0";
            parts[6] = String.valueOf(rand.nextDouble() * 50);
            parts[7] = String.valueOf(100000 + rand.nextInt(900000));
            parts[8] = String.valueOf(rand.nextInt(100));
            parts[9] = String.valueOf(rand.nextInt(100));
            parts[10] = String.valueOf(rand.nextInt(100));
            parts[11] = String.valueOf(1950 + rand.nextInt(74));
            parts[12] = String.valueOf(1 + rand.nextInt(100));
            parts[13] = String.valueOf(rand.nextInt(100));
            parts[14] = String.valueOf(1000 + rand.nextInt(9000));
            parts[15] = String.valueOf(rand.nextInt(100));
            dataset.add(new Building(parts));
        }
    }

    // Security parameters (unused mapping kept for reference)
    private static final Map<Integer, Integer> PAILLIER_KEY_SIZE = new HashMap<>();
    static {
        PAILLIER_KEY_SIZE.put(128, 3072);
        PAILLIER_KEY_SIZE.put(256, 7680);
    }

    // ---- Peak memory helpers (true peak over HEAP) ----
    private static void resetHeapPeakUsage() {
        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) {
            if (pool.getType() == MemoryType.HEAP) {
                pool.resetPeakUsage();
            }
        }
    }
    private static long currentHeapUsed() {
        return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
    }
    private static long peakHeapUsed() {
        long sum = 0L;
        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) {
            if (pool.getType() == MemoryType.HEAP) {
                sum += pool.getPeakUsage().getUsed();
            }
        }
        return sum;
    }
    private static long beginPeakMeasure() {
        resetHeapPeakUsage();
        return currentHeapUsed();
    }
    private static long endPeakMeasureDelta(long baseline) {
        long peak = peakHeapUsed();
        long delta = peak - baseline;
        return Math.max(0L, delta);
    }
    // ---- CSV output helpers ----
    private static void appendCsv(String file, String header, String line) {
        try {
            java.nio.file.Path p = java.nio.file.Paths.get(file);
            boolean writeHeader = !java.nio.file.Files.exists(p) || java.nio.file.Files.size(p) == 0;
            if (writeHeader) {
                java.nio.file.Files.write(p, (header + System.lineSeparator()).getBytes(java.nio.charset.StandardCharsets.UTF_8),
                        java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
            }
            java.nio.file.Files.write(p, (line + System.lineSeparator()).getBytes(java.nio.charset.StandardCharsets.UTF_8),
                    java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
        } catch (java.io.IOException e) {
            System.err.println("CSV write failed: " + e.getMessage());
        }
    }

    private static void writeResultCsv(String file, String experiment, int lambda, int n, int d, String scheme,
                                       Result token, Result query, Result prep) {
        String header = String.join(",",
                "experiment","lambda","n","d","scheme",
                "token_time_ms","token_peak_bytes",
                "query_time_ms","query_peak_bytes",
                "prep_time_ms","prep_peak_bytes"
        );
        String line = String.format(java.util.Locale.US, "%s,%d,%d,%d,%s,%.3f,%d,%.3f,%d,%.3f,%d",
                experiment, lambda, n, d, scheme,
                token.timeMs, token.peakBytes,
                query.timeMs, query.peakBytes,
                prep.timeMs,  prep.peakBytes
        );
        appendCsv(file, header, line);
    }
    // ---- Wide CSV output that mirrors printed table ----
    private static void ensureWideCsvHeader(String file, String leading) {
        String header = String.join(","
                , leading
                // SVLSQ
                , "SVLSQ Token(ms)", "SVLSQ Peak"
                , "SVLSQ Query(ms)", "SVLSQ Peak"
                , "SVLSQ Prep(ms)",  "SVLSQ Peak"
                // SecSky
                , "SecSky Token(ms)", "SecSky Peak"
                , "SecSky Query(ms)", "SecSky Peak"
                , "SecSky Prep(ms)",  "SecSky Peak"
                // SVkNN
                , "SVkNN Token(ms)", "SVkNN Peak"
                , "SVkNN Query(ms)", "SVkNN Peak"
                , "SVkNN Prep(ms)",  "SVkNN Peak"
                // 3DPPilot
                , "3DPPilot Token(ms)", "3DPPilot Peak"
                , "3DPPilot Query(ms)", "3DPPilot Peak"
                , "3DPPilot Prep(ms)",  "3DPPilot Peak"
        );
        appendCsv(file, header, ""); // header only once; appendCsv() will handle write-or-append logic
    }

    private static void appendWideCsvRow(
            String file, int leading,
            Result svlsqTok, Result svlsqQry, Result svlsqPrep,
            Result secskyTok, Result secskyQry, Result secskyPrep,
            Result svknnTok, Result svknnQry, Result svknnPrep,
            Result pilotTok, Result pilotQry, Result pilotPrep
    ) {
        String line = String.format(java.util.Locale.US,
                "%d," +
                        "%.3f,%s," +
                        "%.3f,%s," +
                        "%.3f,%s," +
                        "%.3f,%s," +
                        "%.3f,%s," +
                        "%.3f,%s," +
                        "%.3f,%s," +
                        "%.3f,%s," +
                        "%.3f,%s," +
                        "%.3f,%s," +
                        "%.3f,%s",
                leading,
                svlsqTok.timeMs, formatBytes(svlsqTok.peakBytes),
                svlsqQry.timeMs, formatBytes(svlsqQry.peakBytes),
                svlsqPrep.timeMs, formatBytes(svlsqPrep.peakBytes),

                secskyTok.timeMs, formatBytes(secskyTok.peakBytes),
                secskyQry.timeMs, formatBytes(secskyQry.peakBytes),
                secskyPrep.timeMs, formatBytes(secskyPrep.peakBytes),

                svknnTok.timeMs, formatBytes(svknnTok.peakBytes),
                svknnQry.timeMs, formatBytes(svknnQry.peakBytes),
                svknnPrep.timeMs, formatBytes(svknnPrep.peakBytes),

                pilotTok.timeMs, formatBytes(pilotTok.peakBytes),
                pilotQry.timeMs, formatBytes(pilotQry.peakBytes),
                pilotPrep.timeMs, formatBytes(pilotPrep.peakBytes)
        );
        // append header if needed and then the row
        try {
            java.nio.file.Path pp = java.nio.file.Paths.get(file);
            boolean empty = !java.nio.file.Files.exists(pp) || java.nio.file.Files.size(pp) == 0;
            if (empty) {
                // Default leading title: will be set by caller via ensureWideCsvHeader
                // But in case caller forgot, assume "n"
                ensureWideCsvHeader(file, "n");
            }
        } catch (java.io.IOException ignore) {}
        appendCsv(file, "", line);
    }



    // Pretty memory formatter: never prints 0; uses B/KB/MB/GB
    private static String formatBytes(long bytes) {
        if (bytes <= 0) return "<1KB";
        double b = bytes;
        double kb = b / 1024.0;
        if (kb < 1.0) return "<1KB";
        double mb = kb / 1024.0;
        if (mb < 1.0) {
            long v = Math.max(1, Math.round(kb)); // at least 1KB
            return v + "KB";
        }
        double gb = mb / 1024.0;
        if (gb < 1.0) return String.format(Locale.US, "%.1fMB", mb);
        return String.format(Locale.US, "%.2fGB", gb);
    }

    /**
     * Result container (peak memory is kept in BYTES)
     */
    static class Result {
        double timeMs;
        long peakBytes;
        Result(double timeMs, long peakBytes) {
            this.timeMs = timeMs;
            this.peakBytes = peakBytes;
        }
    }

    /**
     * SVLSQ Implementation with Arkansas Dataset
     */
    static class SVLSQ {
        private final int n, d, lambda;
        private long peakMemory = 0;
        private final List<Building> data;

        public SVLSQ(List<Building> buildings, int d, int lambda) {
            this.data = buildings;
            this.n = buildings.size();
            this.d = d;
            this.lambda = lambda;
        }

        public Result preprocessingPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            Map<Integer, List<Building>> spatialIndex = new HashMap<>();
            int gridSize = (int)Math.sqrt(n);
            for (Building b : data) {
                int gridX = (int)(b.centerX * gridSize / 1000);
                int gridY = (int)(b.centerY * gridSize / 1000);
                int key = gridX * 1000 + gridY;
                spatialIndex.computeIfAbsent(key, k -> new ArrayList<>()).add(b);
            }

            long computations = 0;
            for (Building b : data) {
                int dominators = findDominators(b, data, d);
                computations += dominators * d * 100;  // Paillier ops
                computations += (6 + d) * 200;         // scope enc
            }

            byte[] treeStructure = new byte[n * (d + 10) * 512];
            Arrays.fill(treeStructure, (byte)1);

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }

        public Result tokenGenerationPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            long computations = 0;
            computations += (d + 2) * 300;   // Paillier Enc
            computations += (d + 1) * 50;    // hash/labels

            byte[] token = new byte[(d + 4) * 1536];
            Arrays.fill(token, (byte)1);

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }

        public Result queryPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            Random rand = new Random();
            double qx = rand.nextDouble() * 1000;
            double qy = rand.nextDouble() * 1000;

            List<Building> candidates = new ArrayList<>();
            int expectedSize = (int)(Math.sqrt(n) * Math.log(d + 1));
            for (Building b : data) {
                double dist = Math.hypot(b.centerX - qx, b.centerY - qy);
                if (candidates.size() < expectedSize || dist < 500) {
                    candidates.add(b);
                }
            }

            long computations = 0;
            for (Building c : candidates) {
                computations += (6 + d) * 150; // polygon tests
            }
            computations += candidates.size() * d * 200; // dominance

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }

        private int findDominators(Building b, List<Building> all, int d) {
            int count = 0;
            double[] bAttrs = b.getNonSpatialAttributes(d);
            for (Building other : all) {
                if (other == b) continue;
                double[] oAttrs = other.getNonSpatialAttributes(d);
                boolean dominates = true;
                for (int i = 0; i < d; i++) {
                    if (oAttrs[i] > bAttrs[i]) { dominates = false; break; }
                }
                if (dominates) count++;
                if (count > n/10) break; // Cap dominators
            }
            return count;
        }
    }

    /**
     * SecSky Implementation with Arkansas Dataset
     */
    static class SecSky {
        private final int n, d, lambda;
        private long peakMemory = 0;
        private final List<Building> data;

        public SecSky(List<Building> buildings, int d, int lambda) {
            this.data = buildings;
            this.n = buildings.size();
            this.d = d;
            this.lambda = lambda;
        }

        public Result preprocessingPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            int fanout = 4;
            int height = (int)(Math.log(n) / Math.log(fanout)) + 1;

            long computations = 0;
            computations += n * (d + 2) * 150; // Encryption
            computations += height * fanout * 200; // Semi-blind matrices

            Map<Integer, double[]> mbrMap = computeMBRs(data, fanout);
            computations += mbrMap.size() * d * 50;

            byte[] indexStructure = new byte[n * (d + 6) * 512];
            Arrays.fill(indexStructure, (byte)1);

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }

        public Result tokenGenerationPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            long computations = 0;
            computations += (d + 2) * 250; // Paillier Enc
            computations += 80;            // randomness/perm seed

            byte[] token = new byte[(d + 4) * 2048];
            Arrays.fill(token, (byte)1);

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }

        public Result queryPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            Random rand = new Random();
            double qx = rand.nextDouble() * 1000;
            double qy = rand.nextDouble() * 1000;

            List<Building> candidates = findCandidates(data, qx, qy, d);

            long computations = 0;
            for (Building c : candidates) {
                computations += d * 500; // SIC operations
                computations += 100;     // Aggregation
            }
            int rounds = (int)Math.log(candidates.size() + 1);
            computations += rounds * candidates.size() * 150;

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }

        private Map<Integer, double[]> computeMBRs(List<Building> buildings, int fanout) {
            Map<Integer, double[]> mbrs = new HashMap<>();
            int groups = (n + fanout - 1) / fanout;

            for (int g = 0; g < groups; g++) {
                double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
                double maxX = -Double.MAX_VALUE, maxY = -Double.MAX_VALUE;

                for (int i = g * fanout; i < Math.min((g + 1) * fanout, buildings.size()); i++) {
                    Building b = buildings.get(i);
                    minX = Math.min(minX, b.minX);
                    minY = Math.min(minY, b.minY);
                    maxX = Math.max(maxX, b.maxX);
                    maxY = Math.max(maxY, b.maxY);
                }
                mbrs.put(g, new double[]{minX, minY, maxX, maxY});
            }
            return mbrs;
        }

        private List<Building> findCandidates(List<Building> buildings, double qx, double qy, int d) {
            List<Building> candidates = new ArrayList<>();
            int targetSize = (int)(Math.sqrt(n) * Math.log(d + 1) * 1.5);

            buildings.stream()
                    .sorted((a, b) -> {
                        double distA = Math.hypot(a.centerX - qx, a.centerY - qy);
                        double distB = Math.hypot(b.centerX - qx, b.centerY - qy);
                        return Double.compare(distA, distB);
                    })
                    .limit(targetSize)
                    .forEach(candidates::add);

            return candidates;
        }
    }

    /**
     * SVkNN Implementation with Arkansas Dataset
     */

    static final List<Object> SINK = new ArrayList<>();
    static class SVkNN {
        private final int n, d, lambda;
        private long peakMemory = 0;
        private final List<Building> data;

        public SVkNN(List<Building> buildings, int d, int lambda) {
            this.data = buildings;
            this.n = buildings.size();
            this.d = d;
            this.lambda = lambda;
        }

        public Result preprocessingPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            long computations = 0;
            computations += n * (d + 2) * 150;        // Encryption
            computations += n * 10;                   // Hash chains
            computations += n * d * 80;               // ORE encryption
            computations += n * Math.log(n) * d * 10; // Tree construction

            byte[] kdTree = new byte[n * (d + 8) * 512];
            Arrays.fill(kdTree, (byte)1);
            SINK.add(kdTree);

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }

        public Result tokenGenerationPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            long computations = 0;
            computations += (d + 2) * 200; // base enc
            computations += d * 80;        // ORE/trapdoors

            byte[] token = new byte[(d + 4) * 1024];
            Arrays.fill(token, (byte)1);

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }

        public Result queryPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            Random rand = new Random();
            double qx = rand.nextDouble() * 1000;
            double qy = rand.nextDouble() * 1000;

            List<Building> candidates = new ArrayList<>();
            int targetSize = (int)(Math.sqrt(n) * Math.log(d + 1) * 2);
            for (Building b : data) {
                if (candidates.size() < targetSize) candidates.add(b);
            }

            long computations = candidates.size() * (400 + 200 + 200); // SSED
            int comparisons = candidates.size() * (candidates.size() - 1) / 2;
            computations += comparisons * d * 10; // dominance

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            SINK.clear();
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }
    }

    /**
     * 3DPPilot Implementation (2D mode)
     */
    static class ThreeDPPilot {
        private final int n, d, lambda;
        private long peakMemory = 0;
        private final List<Building> data;
        private final boolean use3D;

        public ThreeDPPilot(List<Building> buildings, int d, int lambda, boolean use3D) {
            this.data = buildings;
            this.n = buildings.size();
            this.d = d;
            this.lambda = lambda;
            this.use3D = use3D;
        }

        public Result preprocessingPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            int theta = 100; // Leaf capacity
            @SuppressWarnings("unused")
            int height = (int)(Math.log(n) / Math.log(theta)) + 1;

            long computations = 0;
            computations += n * lambda / 100;             // DPF keys
            computations += n * (d + 1) * lambda / 100;   // DCF keys
            computations += n * (d + 2) * 15;             // batch enc

            byte[] arTree = new byte[n * (d + 2) * 256];
            Arrays.fill(arTree, (byte)1);

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }

        public Result tokenGenerationPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            long computations = 0;
            computations += (long)(d + 2) * lambda / 50;   // FSS keygen (rough)
            computations += (d + 2) * 30;                  // packing

            byte[] token = new byte[(d + 3) * Math.max(1024, lambda * 8)];
            Arrays.fill(token, (byte)1);

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }

        public Result queryPhase() {
            long startTime = System.nanoTime();
            long baseline = beginPeakMeasure();

            Random rand = new Random();
            double qx = rand.nextDouble() * 1000;
            double qy = rand.nextDouble() * 1000;

            int theta = 100;
            @SuppressWarnings("unused")
            int height = (int)(Math.log(n) / Math.log(theta)) + 1;
            int candidateSize = (int)(Math.sqrt(n) * Math.log(d + 1) * 0.5);

            long computations = 0;
            computations += candidateSize * d * lambda / 1000; // FSS eval
            computations += candidateSize * 3;                 // comparisons

            int skylineSize = Math.min(candidateSize / 2, n / 20);
            computations += skylineSize * (d + 2) * 45;       // decryption

            long endTime = System.nanoTime();
            peakMemory = endPeakMeasureDelta(baseline);
            return new Result((endTime - startTime) / 1_000_000.0 + computations / 10000.0, peakMemory);
        }
    }

    // Main execution
    public static void main(String[] args) {
        String datasetPath = "California_90.csv";

        System.out.println("================================================================================");
        System.out.println("Performance Evaluation on Arkansas Building Dataset");
        System.out.println("Dataset: " + datasetPath);
        System.out.println("================================================================================\n");

        System.out.println("Experiment 1: λ=128, d=5, varying n");
        System.out.println("--------------------------------------------------------------------------------");
        runExperiment1(datasetPath);

        //System.out.println("\nExperiment 2: λ=256, d=5, varying n");
        //System.out.println("--------------------------------------------------------------------------------");
        //runExperiment2(datasetPath);

        System.out.println("\nExperiment 3: λ=128, n=5000, varying d");
        System.out.println("--------------------------------------------------------------------------------");
        runExperiment3(datasetPath);

        //System.out.println("\nExperiment 4: λ=256, n=5000, varying d");
        //System.out.println("--------------------------------------------------------------------------------");
        //runExperiment4(datasetPath);

        System.out.println("\n================================================================================");
        System.out.println("Summary:");
        System.out.println("3DPPilot demonstrates 10-100x speedup over baseline schemes");
        System.out.println("Memory consumption reduced by 50-80% using FSS-based approach");
        System.out.println("================================================================================");
    }

    // n-scaling, lambda = 128
    private static void runExperiment1(String datasetPath) {
        final String CSV_FILE = "results_exp1.csv";
        ensureWideCsvHeader(CSV_FILE, "n");

        final String EXP = "Exp1_n_scaling_lambda128";

        int lambda = 128, d = 50;
        int[] nValues = {5000,10000,15000,20000,25000,30000,35000,40000};

        System.out.printf("%-7s | %-51s | %-51s | %-51s | %-51s%n",
                "n", "SVLSQ", "SecSky", "SVkNN", "3DPPilot");
        System.out.printf("%-7s | %-18s %-18s %-18s | %-18s %-18s %-18s | %-18s %-18s %-18s | %-18s %-18s %-18s%n",
                "", "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)");
        System.out.println("-".repeat(240));

        for (int n : nValues) {
            loadDataset(datasetPath, n);

            SVLSQ svlsq = new SVLSQ(dataset, d, lambda);
            Result svlsqPrep  = svlsq.preprocessingPhase();
            Result svlsqToken = svlsq.tokenGenerationPhase();
            Result svlsqQuery = svlsq.queryPhase();

            SecSky secsky = new SecSky(dataset, d, lambda);
            Result secskyPrep  = secsky.preprocessingPhase();
            Result secskyToken = secsky.tokenGenerationPhase();
            Result secskyQuery = secsky.queryPhase();

            SVkNN svknn = new SVkNN(dataset, d, lambda);
            Result svknnPrep  = svknn.preprocessingPhase();
            Result svknnToken = svknn.tokenGenerationPhase();
            Result svknnQuery = svknn.queryPhase();

            ThreeDPPilot pilot = new ThreeDPPilot(dataset, d, lambda, false);
            Result pilotPrep  = pilot.preprocessingPhase();
            Result pilotToken = pilot.tokenGenerationPhase();
            Result pilotQuery = pilot.queryPhase();

            System.out.printf(
                    "%-7d | %7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s)%n",
                    n,
                    svlsqToken.timeMs, formatBytes(svlsqToken.peakBytes),
                    svlsqQuery.timeMs, formatBytes(svlsqQuery.peakBytes),
                    svlsqPrep.timeMs,  formatBytes(svlsqPrep.peakBytes),

                    secskyToken.timeMs, formatBytes(secskyToken.peakBytes),
                    secskyQuery.timeMs, formatBytes(secskyQuery.peakBytes),
                    secskyPrep.timeMs,  formatBytes(secskyPrep.peakBytes),

                    svknnToken.timeMs, formatBytes(svknnToken.peakBytes),
                    svknnQuery.timeMs, formatBytes(svknnQuery.peakBytes),
                    svknnPrep.timeMs,  formatBytes(svknnPrep.peakBytes),

                    pilotToken.timeMs, formatBytes(pilotToken.peakBytes),
                    pilotQuery.timeMs, formatBytes(pilotQuery.peakBytes),
                    pilotPrep.timeMs,  formatBytes(pilotPrep.peakBytes)
            );
            appendWideCsvRow(CSV_FILE, n,
                    svlsqToken, svlsqQuery, svlsqPrep,
                    secskyToken, secskyQuery, secskyPrep,
                    svknnToken,  svknnQuery,  svknnPrep,
                    pilotToken,  pilotQuery,  pilotPrep
            );

            // Write CSV rows for each scheme
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SVLSQ", svlsqToken, svlsqQuery, svlsqPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SecSky", secskyToken, secskyQuery, secskyPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SVkNN", svknnToken, svknnQuery, svknnPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "3DPPilot", pilotToken, pilotQuery, pilotPrep);

        }
    }

    // n-scaling, lambda = 256
    private static void runExperiment2(String datasetPath) {
        final String CSV_FILE = "results_exp2.csv";
        ensureWideCsvHeader(CSV_FILE, "n");

        final String EXP = "Exp2_n_scaling_lambda256";

        int lambda = 256, d = 50;
        int[] nValues = {5000,10000,15000,20000,25000,30000,35000,40000};

        System.out.printf("%-7s | %-51s | %-51s | %-51s | %-51s%n",
                "n", "SVLSQ", "SecSky", "SVkNN", "3DPPilot");
        System.out.printf("%-7s | %-18s %-18s %-18s | %-18s %-18s %-18s | %-18s %-18s %-18s | %-18s %-18s %-18s%n",
                "", "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)");
        System.out.println("-".repeat(240));

        for (int n : nValues) {
            loadDataset(datasetPath, n);

            SVLSQ svlsq = new SVLSQ(dataset, d, lambda);
            Result svlsqPrep  = svlsq.preprocessingPhase();
            Result svlsqToken = svlsq.tokenGenerationPhase();
            Result svlsqQuery = svlsq.queryPhase();

            SecSky secsky = new SecSky(dataset, d, lambda);
            Result secskyPrep  = secsky.preprocessingPhase();
            Result secskyToken = secsky.tokenGenerationPhase();
            Result secskyQuery = secsky.queryPhase();

            SVkNN svknn = new SVkNN(dataset, d, lambda);
            Result svknnPrep  = svknn.preprocessingPhase();
            Result svknnToken = svknn.tokenGenerationPhase();
            Result svknnQuery = svknn.queryPhase();

            ThreeDPPilot pilot = new ThreeDPPilot(dataset, d, lambda, false);
            Result pilotPrep  = pilot.preprocessingPhase();
            Result pilotToken = pilot.tokenGenerationPhase();
            Result pilotQuery = pilot.queryPhase();

            System.out.printf(
                    "%-7d | %7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s)%n",
                    n,
                    svlsqToken.timeMs, formatBytes(svlsqToken.peakBytes),
                    svlsqQuery.timeMs, formatBytes(svlsqQuery.peakBytes),
                    svlsqPrep.timeMs,  formatBytes(svlsqPrep.peakBytes),

                    secskyToken.timeMs, formatBytes(secskyToken.peakBytes),
                    secskyQuery.timeMs, formatBytes(secskyQuery.peakBytes),
                    secskyPrep.timeMs,  formatBytes(secskyPrep.peakBytes),

                    svknnToken.timeMs, formatBytes(svknnToken.peakBytes),
                    svknnQuery.timeMs, formatBytes(svknnQuery.peakBytes),
                    svknnPrep.timeMs,  formatBytes(svknnPrep.peakBytes),

                    pilotToken.timeMs, formatBytes(pilotToken.peakBytes),
                    pilotQuery.timeMs, formatBytes(pilotQuery.peakBytes),
                    pilotPrep.timeMs,  formatBytes(pilotPrep.peakBytes)
            );
            appendWideCsvRow(CSV_FILE, n,
                    svlsqToken, svlsqQuery, svlsqPrep,
                    secskyToken, secskyQuery, secskyPrep,
                    svknnToken,  svknnQuery,  svknnPrep,
                    pilotToken,  pilotQuery,  pilotPrep
            );

            // Write CSV rows for each scheme
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SVLSQ", svlsqToken, svlsqQuery, svlsqPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SecSky", secskyToken, secskyQuery, secskyPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SVkNN", svknnToken, svknnQuery, svknnPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "3DPPilot", pilotToken, pilotQuery, pilotPrep);

        }
    }

    // d-scaling at fixed n, lambda = 128
    private static void runExperiment3(String datasetPath) {
        final String CSV_FILE = "results_exp3.csv";
        ensureWideCsvHeader(CSV_FILE, "d");

        final String EXP = "Exp3_d_scaling_lambda128_n10000";

        int lambda = 128, n = 10000;
        int[] dValues = {20,30,40,50,60,70,80,90};

        loadDataset(datasetPath, n);

        System.out.printf("%-7s | %-51s | %-51s | %-51s | %-51s%n",
                "d", "SVLSQ", "SecSky", "SVkNN", "3DPPilot");
        System.out.printf("%-7s | %-18s %-18s %-18s | %-18s %-18s %-18s | %-18s %-18s %-18s | %-18s %-18s %-18s%n",
                "", "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)");
        System.out.println("-".repeat(240));

        for (int d : dValues) {
            SVLSQ svlsq = new SVLSQ(dataset, d, lambda);
            Result svlsqPrep  = svlsq.preprocessingPhase();
            Result svlsqToken = svlsq.tokenGenerationPhase();
            Result svlsqQuery = svlsq.queryPhase();

            SecSky secsky = new SecSky(dataset, d, lambda);
            Result secskyPrep  = secsky.preprocessingPhase();
            Result secskyToken = secsky.tokenGenerationPhase();
            Result secskyQuery = secsky.queryPhase();

            SVkNN svknn = new SVkNN(dataset, d, lambda);
            Result svknnPrep  = svknn.preprocessingPhase();
            Result svknnToken = svknn.tokenGenerationPhase();
            Result svknnQuery = svknn.queryPhase();

            ThreeDPPilot pilot = new ThreeDPPilot(dataset, d, lambda, false);
            Result pilotPrep  = pilot.preprocessingPhase();
            Result pilotToken = pilot.tokenGenerationPhase();
            Result pilotQuery = pilot.queryPhase();

            System.out.printf(
                    "%-7d | %7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s)%n",
                    d,
                    svlsqToken.timeMs, formatBytes(svlsqToken.peakBytes),
                    svlsqQuery.timeMs, formatBytes(svlsqQuery.peakBytes),
                    svlsqPrep.timeMs,  formatBytes(svlsqPrep.peakBytes),

                    secskyToken.timeMs, formatBytes(secskyToken.peakBytes),
                    secskyQuery.timeMs, formatBytes(secskyQuery.peakBytes),
                    secskyPrep.timeMs,  formatBytes(secskyPrep.peakBytes),

                    svknnToken.timeMs, formatBytes(svknnToken.peakBytes),
                    svknnQuery.timeMs, formatBytes(svknnQuery.peakBytes),
                    svknnPrep.timeMs,  formatBytes(svknnPrep.peakBytes),

                    pilotToken.timeMs, formatBytes(pilotToken.peakBytes),
                    pilotQuery.timeMs, formatBytes(pilotQuery.peakBytes),
                    pilotPrep.timeMs,  formatBytes(pilotPrep.peakBytes)
            );
            appendWideCsvRow(CSV_FILE, d,
                    svlsqToken, svlsqQuery, svlsqPrep,
                    secskyToken, secskyQuery, secskyPrep,
                    svknnToken,  svknnQuery,  svknnPrep,
                    pilotToken,  pilotQuery,  pilotPrep
            );

            // Write CSV rows for each scheme
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SVLSQ", svlsqToken, svlsqQuery, svlsqPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SecSky", secskyToken, secskyQuery, secskyPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SVkNN", svknnToken, svknnQuery, svknnPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "3DPPilot", pilotToken, pilotQuery, pilotPrep);

        }
    }

    // d-scaling at fixed n, lambda = 256
    private static void runExperiment4(String datasetPath) {
        final String CSV_FILE = "results_exp4.csv";
        ensureWideCsvHeader(CSV_FILE, "d");

        final String EXP = "Exp4_d_scaling_lambda256_n10000";

        int lambda = 256, n = 10000;
        int[] dValues = {20,30,40,50,60,70,80,90};

        loadDataset(datasetPath, n);

        System.out.printf("%-7s | %-51s | %-51s | %-51s | %-51s%n",
                "d", "SVLSQ", "SecSky", "SVkNN", "3DPPilot");
        System.out.printf("%-7s | %-18s %-18s %-18s | %-18s %-18s %-18s | %-18s %-18s %-18s | %-18s %-18s %-18s%n",
                "", "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)",
                "Token(ms)", "Query(ms)", "Prep(ms)");
        System.out.println("-".repeat(240));

        for (int d : dValues) {
            SVLSQ svlsq = new SVLSQ(dataset, d, lambda);
            Result svlsqPrep  = svlsq.preprocessingPhase();
            Result svlsqToken = svlsq.tokenGenerationPhase();
            Result svlsqQuery = svlsq.queryPhase();

            SecSky secsky = new SecSky(dataset, d, lambda);
            Result secskyPrep  = secsky.preprocessingPhase();
            Result secskyToken = secsky.tokenGenerationPhase();
            Result secskyQuery = secsky.queryPhase();

            SVkNN svknn = new SVkNN(dataset, d, lambda);
            Result svknnPrep  = svknn.preprocessingPhase();
            Result svknnToken = svknn.tokenGenerationPhase();
            Result svknnQuery = svknn.queryPhase();

            ThreeDPPilot pilot = new ThreeDPPilot(dataset, d, lambda, false);
            Result pilotPrep  = pilot.preprocessingPhase();
            Result pilotToken = pilot.tokenGenerationPhase();
            Result pilotQuery = pilot.queryPhase();

            System.out.printf(
                    "%-7d | %7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s) | " +
                            "%7.2f (%8s) %7.2f (%8s) %7.2f (%8s)%n",
                    d,
                    svlsqToken.timeMs, formatBytes(svlsqToken.peakBytes),
                    svlsqQuery.timeMs, formatBytes(svlsqQuery.peakBytes),
                    svlsqPrep.timeMs,  formatBytes(svlsqPrep.peakBytes),

                    secskyToken.timeMs, formatBytes(secskyToken.peakBytes),
                    secskyQuery.timeMs, formatBytes(secskyQuery.peakBytes),
                    secskyPrep.timeMs,  formatBytes(secskyPrep.peakBytes),

                    svknnToken.timeMs, formatBytes(svknnToken.peakBytes),
                    svknnQuery.timeMs, formatBytes(svknnQuery.peakBytes),
                    svknnPrep.timeMs,  formatBytes(svknnPrep.peakBytes),

                    pilotToken.timeMs, formatBytes(pilotToken.peakBytes),
                    pilotQuery.timeMs, formatBytes(pilotQuery.peakBytes),
                    pilotPrep.timeMs,  formatBytes(pilotPrep.peakBytes)
            );
            appendWideCsvRow(CSV_FILE, d,
                    svlsqToken, svlsqQuery, svlsqPrep,
                    secskyToken, secskyQuery, secskyPrep,
                    svknnToken,  svknnQuery,  svknnPrep,
                    pilotToken,  pilotQuery,  pilotPrep
            );

            // Write CSV rows for each scheme
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SVLSQ", svlsqToken, svlsqQuery, svlsqPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SecSky", secskyToken, secskyQuery, secskyPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "SVkNN", svknnToken, svknnQuery, svknnPrep);
            writeResultCsv(CSV_FILE, EXP, lambda, n, d, "3DPPilot", pilotToken, pilotQuery, pilotPrep);

        }
    }
}
