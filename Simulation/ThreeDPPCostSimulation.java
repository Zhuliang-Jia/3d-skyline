import java.io.*;
import java.util.*;
import java.security.SecureRandom;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;

/**
 * 3DPPilot: Privacy-Preserving Location-Based Skyline Queries in 3D Obstacle Space
 * Stage-level timing + incremental peak-heap memory per stage
 */
public class ThreeDPPCostSimulation {

    // Configuration parameters
    private int n; // Number of buildings
    private int d; // Number of non-spatial attributes
    private int lambda; // Security parameter (128 or 256 bits)
    private int numObstacles; // Number of 3D obstacles
    private int numLandmarks; // Number of landmarks per region
    private String inputFile; // CSV input file name
    private int theta; // leaf capacity (theta)
    private int c; // max children per internal node

    // Data structures
    private List<Building> dataset;
    private List<Obstacle> obstacles;
    private ARTree arTree;
    private SecretKey encryptionKey;
    private SecureRandom random;

    // Detailed timing measurements
    private PhaseTimings phaseTimings;
    // Stage-level stats (time + peak memory)
    private StageStats stageStats;

    /** Participant-level timing container */
    static class ParticipantTiming {
        long dataOwner = 0;
        long cs0 = 0;
        long cs1 = 0;
        long queryUser = 0;
        long getTotal() {
            return dataOwner + cs0 + cs1 + queryUser;
        }
        void add(ParticipantTiming other) {
            this.dataOwner += other.dataOwner;
            this.cs0 += other.cs0;
            this.cs1 += other.cs1;
            this.queryUser += other.queryUser;
        }
    }

    /** Phase-level timings (kept for compatibility) */
    static class PhaseTimings {
        ParticipantTiming preprocessing; // (System Init + Data Upload)
        ParticipantTiming query;         // (Token Gen + Secure Query + Retrieval)
        PhaseTimings() {
            preprocessing = new ParticipantTiming();
            query = new ParticipantTiming();
        }
    }

    /** Stage-level metrics: aggregated across participants, with peak heap memory per stage. */
    static class StageMetric {
        long timeNanos = 0;
        long peakBytes = 0;
    }
    static class StageStats {
        StageMetric dataUpload = new StageMetric();
        StageMetric tokenGeneration = new StageMetric();
        StageMetric secureQuery = new StageMetric();
    }

    static final int PER_DIM_ROUNDS = 32;

    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        return z ^ (z >>> 31);
    }

    // ===== Memory helpers =====
    private void resetHeapPeaks() {
        try {
            for (java.lang.management.MemoryPoolMXBean pool : java.lang.management.ManagementFactory.getMemoryPoolMXBeans()) {
                if (pool.getType() == java.lang.management.MemoryType.HEAP) {
                    pool.resetPeakUsage();
                }
            }
        } catch (Exception ignore) {}
    }

    // Sum of peak used across all HEAP pools (for better consistency with baseline below)
    private long getHeapPeakBytes() {
        long peak = 0;
        try {
            for (java.lang.management.MemoryPoolMXBean pool : java.lang.management.ManagementFactory.getMemoryPoolMXBeans()) {
                if (pool.getType() == java.lang.management.MemoryType.HEAP) {
                    java.lang.management.MemoryUsage mu = pool.getPeakUsage();
                    if (mu != null) peak += mu.getUsed();
                }
            }
        } catch (Exception ignore) {}
        return peak;
    }

    // Current HEAP usage baseline: sum across all HEAP pools
    private long heapUsedNow() {
        long used = 0;
        try {
            for (java.lang.management.MemoryPoolMXBean pool : java.lang.management.ManagementFactory.getMemoryPoolMXBeans()) {
                if (pool.getType() == java.lang.management.MemoryType.HEAP) {
                    java.lang.management.MemoryUsage mu = pool.getUsage();
                    if (mu != null) used += mu.getUsed();
                }
            }
        } catch (Exception ignore) {}
        return used;
    }

    // Constructors
    public ThreeDPPCostSimulation(int n, int d, int lambda, int numObstacles,
                                  int numLandmarks, String inputFile) {
        this.n = n;
        this.d = d;
        this.lambda = lambda;
        this.numObstacles = numObstacles;
        this.numLandmarks = numLandmarks;
        this.inputFile = inputFile;
        this.dataset = new ArrayList<>();
        this.obstacles = new ArrayList<>();
        this.arTree = new ARTree();
        this.stageStats = new StageStats();
        this.theta = 100;
        this.c = 2; // defaults
        this.random = new SecureRandom();
        this.phaseTimings = new PhaseTimings();
        this.random.setSeed(123456789L);
    }
    public ThreeDPPCostSimulation(int n, int d, int lambda, int numObstacles,
                                  int numLandmarks, int theta, int c, String inputFile) {
        this(n, d, lambda, numObstacles, numLandmarks, inputFile);
        this.theta = theta;
        this.c = c;
    }

    // Building representation
    static class Building {
        int buildingId;
        double minX, maxX, minY, maxY, minZ, maxZ;
        double centerX, centerY, centerZ;
        double[] nonSpatialAttrs;
        Building(int id, double minX, double maxX, double minY, double maxY, double minZ, double maxZ, double[] nonSpatialAttrs) {
            this.buildingId = id;
            this.minX = minX;
            this.maxX = maxX;
            this.minY = minY;
            this.maxY = maxY;
            this.minZ = minZ;
            this.maxZ = maxZ;
            this.centerX = (minX + maxX) / 2;
            this.centerY = (minY + maxY) / 2;
            this.centerZ = (minZ + maxZ) / 2;
            this.nonSpatialAttrs = nonSpatialAttrs;
        }
        double getAttr(int idx) {
            return nonSpatialAttrs == null || idx >= nonSpatialAttrs.length ? 0 : nonSpatialAttrs[idx];
        }
    }

    // Obstacle representation (simplified)
    static class Obstacle {
        double x, y, z, r;
        Obstacle(double x, double y, double z, double r) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.r = r;
        }
    }

    // AR-Tree node
    class ARTreeNode {
        boolean isLeaf;
        List<Building> objects;
        List<ARTreeNode> children;
        double[] minBounds;
        double[] maxBounds;
        double[] attrBounds;
        double distMin, distMax;
        byte[] encryptedData;
        List<double[]> landmarks;
        ARTreeNode(boolean isLeaf) {
            this.isLeaf = isLeaf;
            this.objects = new ArrayList<>();
            this.children = new ArrayList<>();
            this.landmarks = new ArrayList<>();
        }
    }

    // AR-Tree structure
    class ARTree {
        ARTreeNode root;
        int maxLeafSize = 50;
        int branchingFactor = 2;
        double[] origin = {0,0,0};
        ARTree() { this.root = new ARTreeNode(false); }
        void build(List<Building> buildings, int numLandmarks) {
            root = buildNode(buildings, 0, numLandmarks);
            computeDistanceBounds(root);
        }
        private ARTreeNode buildNode(List<Building> buildings, int depth, int numLandmarks) {
            if (buildings.size() <= maxLeafSize) {
                ARTreeNode leaf = new ARTreeNode(true);
                leaf.objects = new ArrayList<>(buildings);
                computeBounds(leaf);
                generateLandmarks(leaf, numLandmarks);
                return leaf;
            }
            ARTreeNode internal = new ARTreeNode(false);
            List<List<Building>> splits = spatialSplit(buildings, depth);
            for (List<Building> split : splits) if (!split.isEmpty()) internal.children.add(buildNode(split, depth + 1, numLandmarks));
            computeBounds(internal);
            return internal;
        }
        private List<List<Building>> spatialSplit(List<Building> buildings, int depth) {
            int splitDim = depth % 3;
            buildings.sort((a,b)->{
                double aVal = (splitDim==0? a.centerX : splitDim==1? a.centerY : a.centerZ);
                double bVal = (splitDim==0? b.centerX : splitDim==1? b.centerY : b.centerZ);
                return Double.compare(aVal,bVal);
            });
            int cLocal = Math.max(2, branchingFactor);
            int size = buildings.size();
            int base = size / cLocal;
            int rem = size % cLocal;
            int idx = 0;
            List<List<Building>> result = new ArrayList<>();
            for (int k=0;k<cLocal;k++) {
                int partSize = base + (k<rem?1:0);
                int nextIdx = Math.min(size, idx + partSize);
                if (nextIdx>idx) result.add(new ArrayList<>(buildings.subList(idx, nextIdx)));
                idx = nextIdx;
            }
            return result;
        }
        private void computeBounds(ARTreeNode node) {
            if (node.isLeaf && !node.objects.isEmpty()) {
                node.minBounds = new double[]{
                        node.objects.stream().mapToDouble(b->b.minX).min().orElse(0), node.objects.stream().mapToDouble(b->b.minY).min().orElse(0), node.objects.stream().mapToDouble(b->b.minZ).min().orElse(0)
                };
                node.maxBounds = new double[]{
                        node.objects.stream().mapToDouble(b->b.maxX).max().orElse(0), node.objects.stream().mapToDouble(b->b.maxY).max().orElse(0), node.objects.stream().mapToDouble(b->b.maxZ).max().orElse(0)
                };
                if (node.objects.get(0).nonSpatialAttrs != null) {
                    int numAttrs = node.objects.get(0).nonSpatialAttrs.length;
                    node.attrBounds = new double[numAttrs];
                    for (int i=0;i<numAttrs;i++){
                        final int idx=i;
                        node.attrBounds[i]= node.objects.stream().mapToDouble(b->b.getAttr(idx)).max().orElse(0);
                    }
                }
            } else if (!node.isLeaf) {
                node.minBounds = new double[3];
                node.maxBounds = new double[3];
                Arrays.fill(node.minBounds, Double.POSITIVE_INFINITY);
                Arrays.fill(node.maxBounds, Double.NEGATIVE_INFINITY);
                for (ARTreeNode child: node.children){
                    computeBounds(child);
                    for(int i=0;i<3;i++){
                        node.minBounds[i]=Math.min(node.minBounds[i],child.minBounds[i]);
                        node.maxBounds[i]=Math.max(node.maxBounds[i],child.maxBounds[i]);
                    }
                }
            }
        }
        private void computeDistanceBounds(ARTreeNode node) {
            if (node == null) return;
            double centerX=(node.minBounds[0]+node.maxBounds[0])/2, centerY=(node.minBounds[1]+node.maxBounds[1])/2, centerZ=(node.minBounds[2]+node.maxBounds[2])/2;
            double halfDiag = Math.sqrt(Math.pow((node.maxBounds[0]-node.minBounds[0])/2,2)+Math.pow((node.maxBounds[1]-node.minBounds[1])/2,2)+Math.pow((node.maxBounds[2]-node.minBounds[2])/2,2));
            double centerDist = Math.sqrt(centerX*centerX + centerY*centerY + centerZ*centerZ);
            node.distMin = Math.max(0, centerDist - halfDiag);
            node.distMax = centerDist + halfDiag;
            if (!node.isLeaf) for (ARTreeNode child: node.children) computeDistanceBounds(child);
        }
    }

    // Utility: generate landmarks for a leaf node
    private void generateLandmarks(ARTreeNode node, int L) {
        node.landmarks.clear();
        for (int i=0;i<L;i++) node.landmarks.add(new double[]{
                node.minBounds[0]+random.nextDouble()*(node.maxBounds[0]-node.minBounds[0]), node.minBounds[1]+random.nextDouble()*(node.maxBounds[1]-node.minBounds[1]), node.minBounds[2]+random.nextDouble()*(node.maxBounds[2]-node.minBounds[2])
        });
    }

    // Generate synthetic obstacles (not heavy, kept for completeness)
    private void generateObstacles() {
        obstacles.clear();
        for (int i=0;i<numObstacles;i++) obstacles.add(new Obstacle(random.nextDouble()*100, random.nextDouble()*100, random.nextDouble()*50, 1+random.nextDouble()*3));
    }

    // CSV loader (simplified fallback to synthetic generation)
    private void loadDataFromCSV() {
        dataset.clear();
        try (BufferedReader br = new BufferedReader(new FileReader(inputFile))) {
            String line; int id = 0;
            while ((line = br.readLine()) != null && id < n) {
                String[] p = line.split(",");
                double minX = Double.parseDouble(p[0]);
                double minY = Double.parseDouble(p[1]);
                double minZ = Double.parseDouble(p[2]);
                double maxX = minX + 1 + random.nextDouble()*2;
                double maxY = minY + 1 + random.nextDouble()*2;
                double maxZ = minZ + 1 + random.nextDouble()*2;

                int totalAvailable = Math.max(0, p.length - 3);
                int use = Math.min(d, totalAvailable);
                double[] attrs = new double[d];

                for (int i = 0; i < use; i++) {
                    attrs[i] = Double.parseDouble(p[3 + i]);
                }

                for (int i = use; i < d; i++) {
                    attrs[i] = 0.0;
                }
                dataset.add(new Building(id++, minX, maxX, minY, maxY, minZ, maxZ, attrs));
            }
        } catch (Exception e) {
            generateSyntheticData();
        }
    }

    private void generateSyntheticData() {
        dataset.clear();
        for (int i=0;i<n;i++){
            double minX=random.nextDouble()*100, minY=random.nextDouble()*100, minZ=random.nextDouble()*50;
            double maxX=minX+1+random.nextDouble()*2, maxY=minY+1+random.nextDouble()*2, maxZ=minZ+1+random.nextDouble()*2;
            double[] attrs = new double[d];
            for (int j=0;j<d;j++) attrs[j]=random.nextDouble();
            dataset.add(new Building(i, minX, maxX, minY, maxY, minZ, maxZ, attrs));
        }
    }

    // Secret share creation (mocked)
    private void createSecretShares(ARTreeNode node) {
        if (node == null) return;
        node.encryptedData = new byte[lambda / 8];
        random.nextBytes(node.encryptedData);
        for (ARTreeNode child : node.children) createSecretShares(child);
    }

    // Store tree shares to servers (mocked)
    private void storeTreeShare(ARTreeNode node, boolean cs0) {
        if (node == null) return;
        byte[] payload = Arrays.copyOf(node.encryptedData, node.encryptedData.length);
        for (int i=0;i<5;i++) payload[0] ^= i; // simulate small IO/cpu
        for (ARTreeNode child : node.children) storeTreeShare(child, cs0);
    }

    // Preprocessing Phase (System Init + Data Upload)
    public void preprocessingPhase() {
        ParticipantTiming initTiming = new ParticipantTiming();
        ParticipantTiming uploadTiming = new ParticipantTiming();

        long startTime;
        // === SYSTEM INITIALIZATION ===
        startTime = System.nanoTime();
        try {
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(lambda);
            encryptionKey = keyGen.generateKey();
            int numDPFKeys = n;
            int numDCFKeys = n * (d + 1);
            for (int i=0;i<numDPFKeys;i++){
                byte[] key = new byte[lambda/8];
                random.nextBytes(key);
            }
            for (int i=0;i<numDCFKeys;i++){
                byte[] key = new byte[lambda/8];
                random.nextBytes(key);
            }
            byte[] heParams = new byte[256];
            random.nextBytes(heParams);
            startTime = System.nanoTime();
            byte[] quKeys = new byte[32];
            random.nextBytes(quKeys);
            initTiming.queryUser = System.nanoTime() - startTime;
        } catch (Exception e) {
            e.printStackTrace();
        }
        initTiming.dataOwner = System.nanoTime() - startTime;

        startTime = System.nanoTime();
        byte[] cs0Keys = new byte[(n + n * (d + 1)) * (lambda / 8)];
        random.nextBytes(cs0Keys); initTiming.cs0 = System.nanoTime() - startTime;
        startTime = System.nanoTime();
        byte[] cs1Keys = new byte[(n + n * (d + 1)) * (lambda / 8)];
        random.nextBytes(cs1Keys); initTiming.cs1 = System.nanoTime() - startTime;

        // === DATA UPLOAD ===
        this.arTree.maxLeafSize = this.theta;
        this.arTree.branchingFactor = this.c;
        resetHeapPeaks();
        long baseUsedDU = heapUsedNow();
        long prePeakDU = getHeapPeakBytes(); 
        long stageStartDU = System.nanoTime();

        // Data Owner loads and processes data
        startTime = System.nanoTime();
        loadDataFromCSV();
        if (!dataset.isEmpty()) {
            startTime = System.nanoTime();
            arTree.build(dataset, numLandmarks);
            for (int i=0;i<n;i++){
                byte[] buf = new byte[lambda/8];
                random.nextBytes(buf);
            }
            startTime = System.nanoTime();
            computeVisibilityGraphWithObstacles();
            createSecretShares(arTree.root);
        }
        uploadTiming.dataOwner = System.nanoTime() - startTime;

        startTime = System.nanoTime(); storeTreeShare(arTree.root, true);
        uploadTiming.cs0 = System.nanoTime() - startTime;
        startTime = System.nanoTime(); storeTreeShare(arTree.root, false);
        uploadTiming.cs1 = System.nanoTime() - startTime;

        stageStats.dataUpload.timeNanos = System.nanoTime() - stageStartDU;
        stageStats.dataUpload.peakBytes = Math.max(0L, getHeapPeakBytes() - prePeakDU);

        phaseTimings.preprocessing.add(initTiming);
        phaseTimings.preprocessing.add(uploadTiming);
    }

    // Query Phase (Token Gen + Query + Result Retrieval)
    public void queryPhase() {
        ParticipantTiming tokenTiming = new ParticipantTiming();
        ParticipantTiming queryTiming = new ParticipantTiming();
        ParticipantTiming retrievalTiming = new ParticipantTiming();

        // === TOKEN GENERATION ===
        resetHeapPeaks();
        long baseUsedToken = heapUsedNow();
        long prePeakToken = getHeapPeakBytes();
        long stageStartToken = System.nanoTime();
        long startTime;

        startTime = System.nanoTime();
        double qx=50, qy=50, qz=25;
        if (!dataset.isEmpty()){
            Building b = dataset.get(0);
            qx=b.centerX; qy=b.centerY; qz=b.centerZ;
        }
        try { Cipher cipher = Cipher.getInstance("AES");
            cipher.init(Cipher.ENCRYPT_MODE, encryptionKey);
            byte[] tokenSeed=(qx+","+qy+","+qz).getBytes();
            byte[] encToken=cipher.doFinal(tokenSeed);
        }
        catch(Exception e){
            e.printStackTrace();
        }
        double distToOrigin = Math.sqrt(qx*qx + qy*qy + qz*qz);
        byte[] tokenShare0 = new byte[lambda/8];
        byte[] tokenShare1 = new byte[lambda/8];
        random.nextBytes(tokenShare0);
        long distBits = Double.doubleToLongBits(distToOrigin);
        for (int i=0;i<8;i++) tokenShare1[i]= (byte)(((distBits>>(i*8)) & 0xFF) ^ tokenShare0[i]);
        tokenTiming.queryUser = System.nanoTime() - startTime;

        startTime = System.nanoTime(); byte[] cs0Token = Arrays.copyOf(tokenShare0, tokenShare0.length);
        tokenTiming.cs0 = System.nanoTime() - startTime;
        startTime = System.nanoTime(); byte[] cs1Token = Arrays.copyOf(tokenShare1, tokenShare1.length);
        tokenTiming.cs1 = System.nanoTime() - startTime;

        stageStats.tokenGeneration.timeNanos = (tokenTiming.queryUser + tokenTiming.cs0 + tokenTiming.cs1);
        stageStats.tokenGeneration.peakBytes = Math.max(0L, getHeapPeakBytes() - prePeakToken);

        // === SECURE SKYLINE QUERY ===
        resetHeapPeaks();
        long baseUsedSecure = getHeapPeakBytes();
        long prePeakSecure = getHeapPeakBytes();
        long stageStartSecure = System.nanoTime();

        if (!dataset.isEmpty()) {
            startTime = System.nanoTime(); executeSecureQuery(true);
            queryTiming.cs0 = System.nanoTime() - startTime;
            startTime = System.nanoTime(); executeSecureQuery(false);
            queryTiming.cs1 = System.nanoTime() - startTime;
        }

        stageStats.secureQuery.timeNanos = (queryTiming.cs0 + queryTiming.cs1);
        stageStats.secureQuery.peakBytes = Math.max(0L, getHeapPeakBytes() - baseUsedSecure);

        // === RESULT RETRIEVAL (not part of stage stats) ===
        int skylineSize = Math.min((int)(Math.sqrt(n) * Math.log(d + 1)), n / 10);
        startTime = System.nanoTime();
        byte[][] cs0Results = new byte[skylineSize][];
        for(int i=0;i<skylineSize;i++){
            cs0Results[i]=new byte[64+d*8];
            random.nextBytes(cs0Results[i]);
        }
        retrievalTiming.cs0 = System.nanoTime() - startTime;
        startTime = System.nanoTime();
        byte[][] cs1Results = new byte[skylineSize][];
        for(int i=0;i<skylineSize;i++){
            cs1Results[i]=new byte[64+d*8];
            random.nextBytes(cs1Results[i]);
        }
        retrievalTiming.cs1 = System.nanoTime() - startTime;
        startTime = System.nanoTime();
        for(int i=0;i<skylineSize;i++){
            byte[] reconstructed=new byte[cs0Results[i].length];
            for(int j=0;j<reconstructed.length;j++){
                reconstructed[j]=(byte)(cs0Results[i][j]^cs1Results[i][j]);
            }
        }
        retrievalTiming.queryUser = System.nanoTime() - startTime;

        phaseTimings.query.add(tokenTiming);
        phaseTimings.query.add(queryTiming);
        phaseTimings.query.add(retrievalTiming);
    }

    // Skyline query execution (mock)
    private void executeSecureQuery(boolean cs0) {

        Deque<ARTreeNode> queue = new ArrayDeque<>();
        queue.offer(arTree.root);
        while(!queue.isEmpty()){
            ARTreeNode node = queue.poll();
            if (node.isLeaf) {
                for (Building b : node.objects) {
                    long acc = 0L;
                    for (int i = 0; i < d; i++) {
                        double val=b.getAttr(i);
                        long x = Double.doubleToRawLongBits(val);
                        if (val>0.5) {
                            x = mix64(x);
                            acc ^= x;
                        }
                    }
                }
            } else {
                for (ARTreeNode child : node.children) {
                    boolean pruned = false;
                    for (double[] lm : node.landmarks) {
                        if (random.nextDouble()<0.3) {
                            pruned=true;
                            break;
                        }
                    }
                    if (!pruned) queue.offer(child);
                }
            }
        }
    }

    // Visibility graph with obstacles (mock)
    private void computeVisibilityGraphWithObstacles() {
        for (Obstacle obs : obstacles) {
            double cost = Math.sqrt(obs.x*obs.x + obs.y*obs.y + obs.z*obs.z) * obs.r;
            if (cost>0) {
                continue;
            }
        }
    }

    public void runSimulation() {
        preprocessingPhase();
        queryPhase();
    }

    // ===== Cases 1-4 =====
    public static void case_vary_d_lambda_128(String csvFile) {
        System.out.println("\n================================================================================");
        System.out.println("Case: n=10000, λ=128, varying d (20..90)");
        System.out.println("================================================================================");
        int n = 10000;
        int lambda = 128;
        int[] dValues = {20,30,40,50,60,70,80,90};
        int obstacles = 50;
        int theta = 100;
        int c = 2;
        int iterations = 20;
        int[] landmarksArr = {3};
        String out = "case_n10000_lambda128_vary_d.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(out))) {
            pw.println("case,n,d,lambda,theta,c,landmarks,stage,time_ms,peak_mb");
            System.out.println("d\tTokenGen(ms)\tTokenGen(MB)\tSecureQuery(ms)\tSecureQuery(MB)\tDataUpload(ms)\tDataUpload(MB)");
            for (int d : dValues) {
                long tToken=0, tQuery=0, tUpload=0; long mToken=0, mQuery=0, mUpload=0;
                for (int it=0; it<iterations; it++) {
                    ThreeDPPCostSimulation sim = new ThreeDPPCostSimulation(n, d, lambda, obstacles, landmarksArr[0], theta, c, csvFile);
                    sim.runSimulation();
                    tUpload += sim.stageStats.dataUpload.timeNanos;
                    tToken += sim.stageStats.tokenGeneration.timeNanos;
                    tQuery += sim.stageStats.secureQuery.timeNanos;
                    mUpload = Math.max(mUpload, sim.stageStats.dataUpload.peakBytes);
                    mToken = Math.max(mToken, sim.stageStats.tokenGeneration.peakBytes);
                    mQuery = Math.max(mQuery, sim.stageStats.secureQuery.peakBytes);
                }
                double avgUploadMs = tUpload/(double)iterations/1_000_000.0;
                double avgTokenMs = tToken/(double)iterations/1_000_000.0;
                double avgQueryMs = tQuery/(double)iterations/1_000_000.0;
                long peakUploadMB = mUpload/(1024*1024);
                long peakTokenMB = mToken/(1024*1024);
                long peakQueryMB = mQuery/(1024*1024);
                System.out.printf("%d\t%.3f\t\t%d\t\t%.3f\t\t%d\t\t%.3f\t\t%d\n", d, avgTokenMs, peakTokenMB, avgQueryMs, peakQueryMB, avgUploadMs, peakUploadMB);
                pw.printf("1,%d,%d,%d,%d,%d,%d,TokenGen,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgTokenMs, peakTokenMB);
                pw.printf("1,%d,%d,%d,%d,%d,%d,SecureQuery,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgQueryMs, peakQueryMB);
                pw.printf("1,%d,%d,%d,%d,%d,%d,DataUpload,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgUploadMs, peakUploadMB);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void case_vary_d_lambda_256(String csvFile) {
        System.out.println("\n================================================================================");
        System.out.println("Case: n=10000, λ=256, varying d (20..90)");
        System.out.println("================================================================================");
        int n = 10000;
        int lambda = 256;
        int[] dValues = {20,30,40,50,60,70,80,90};
        int obstacles = 50;
        int theta = 100; int c = 2;
        int iterations = 20;
        int[] landmarksArr = {3};
        String out = "case_n10000_lambda256_vary_d.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(out))) {
            pw.println("case,n,d,lambda,theta,c,landmarks,stage,time_ms,peak_mb");
            System.out.println("d\tTokenGen(ms)\tTokenGen(MB)\tSecureQuery(ms)\tSecureQuery(MB)\tDataUpload(ms)\tDataUpload(MB)");
            for (int d : dValues) {
                long tToken=0, tQuery=0, tUpload=0; long mToken=0, mQuery=0, mUpload=0;
                for (int it=0; it<iterations; it++) {
                    ThreeDPPCostSimulation sim = new ThreeDPPCostSimulation(n, d, lambda, obstacles, landmarksArr[0], theta, c, csvFile);
                    sim.runSimulation();
                    tUpload += sim.stageStats.dataUpload.timeNanos;
                    tToken += sim.stageStats.tokenGeneration.timeNanos;
                    tQuery += sim.stageStats.secureQuery.timeNanos;
                    mUpload = Math.max(mUpload, sim.stageStats.dataUpload.peakBytes);
                    mToken = Math.max(mToken, sim.stageStats.tokenGeneration.peakBytes);
                    mQuery = Math.max(mQuery, sim.stageStats.secureQuery.peakBytes);
                }
                double avgUploadMs = tUpload/(double)iterations/1_000_000.0;
                double avgTokenMs = tToken/(double)iterations/1_000_000.0;
                double avgQueryMs = tQuery/(double)iterations/1_000_000.0;
                long peakUploadMB = mUpload/(1024*1024); long peakTokenMB = mToken/(1024*1024); long peakQueryMB = mQuery/(1024*1024);
                System.out.printf("%d\t%.3f\t\t%d\t\t%.3f\t\t%d\t\t%.3f\t\t%d\n", d, avgTokenMs, peakTokenMB, avgQueryMs, peakQueryMB, avgUploadMs, peakUploadMB);
                pw.printf("2,%d,%d,%d,%d,%d,%d,TokenGen,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgTokenMs, peakTokenMB);
                pw.printf("2,%d,%d,%d,%d,%d,%d,SecureQuery,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgQueryMs, peakQueryMB);
                pw.printf("2,%d,%d,%d,%d,%d,%d,DataUpload,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgUploadMs, peakUploadMB);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void case_vary_n_lambda_128(String csvFile) {
        System.out.println("\n================================================================================");
        System.out.println("Case: d=5, λ=128, varying n (5000..40000)");
        System.out.println("================================================================================");
        int d = 20;
        int lambda = 128;
        int[] nValues = {5000,10000,15000,20000,25000,30000,35000,40000};
        int obstacles = 50;
        int theta = 100;
        int c = 2;
        int iterations = 20;
        int[] landmarksArr = {3};
        String out = "case_d5_lambda128_vary_n.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(out))) {
            pw.println("case,n,d,lambda,theta,c,landmarks,stage,time_ms,peak_mb");
            System.out.println("n\tTokenGen(ms)\tTokenGen(MB)\tSecureQuery(ms)\tSecureQuery(MB)\tDataUpload(ms)\tDataUpload(MB)");
            for (int n : nValues) {
                long tToken=0, tQuery=0, tUpload=0; long mToken=0, mQuery=0, mUpload=0;
                for (int it=0; it<iterations; it++) {
                    ThreeDPPCostSimulation sim = new ThreeDPPCostSimulation(n, d, lambda, obstacles, landmarksArr[0], theta, c, csvFile);
                    sim.runSimulation();
                    tUpload += sim.stageStats.dataUpload.timeNanos;
                    tToken += sim.stageStats.tokenGeneration.timeNanos;
                    tQuery += sim.stageStats.secureQuery.timeNanos;
                    mUpload = Math.max(mUpload, sim.stageStats.dataUpload.peakBytes);
                    mToken = Math.max(mToken, sim.stageStats.tokenGeneration.peakBytes);
                    mQuery = Math.max(mQuery, sim.stageStats.secureQuery.peakBytes);
                }
                double avgUploadMs = tUpload/(double)iterations/1_000_000.0;
                double avgTokenMs = tToken/(double)iterations/1_000_000.0;
                double avgQueryMs = tQuery/(double)iterations/1_000_000.0;
                long peakUploadMB = mUpload/(1024*1024);
                long peakTokenMB = mToken/(1024*1024);
                long peakQueryMB = mQuery/(1024*1024);
                System.out.printf("%d\t%.3f\t\t%d\t\t%.3f\t\t%d\t\t%.3f\t\t%d\n", n, avgTokenMs, peakTokenMB, avgQueryMs, peakQueryMB, avgUploadMs, peakUploadMB);
                pw.printf("3,%d,%d,%d,%d,%d,%d,TokenGen,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgTokenMs, peakTokenMB);
                pw.printf("3,%d,%d,%d,%d,%d,%d,SecureQuery,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgQueryMs, peakQueryMB);
                pw.printf("3,%d,%d,%d,%d,%d,%d,DataUpload,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgUploadMs, peakUploadMB);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void case_vary_n_lambda_256(String csvFile) {
        System.out.println("\n================================================================================");
        System.out.println("Case: d=5, λ=256, varying n (5000..40000)");
        System.out.println("================================================================================");
        int d = 20;
        int lambda = 256;
        int[] nValues = {5000,10000,15000,20000,25000,30000,35000,40000};
        int obstacles = 50;
        int theta = 100;
        int c = 2;
        int iterations = 20;
        int[] landmarksArr = {3};
        String out = "case_d5_lambda256_vary_n.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(out))) {
            pw.println("case,n,d,lambda,theta,c,landmarks,stage,time_ms,peak_mb");
            System.out.println("n\tTokenGen(ms)\tTokenGen(MB)\tSecureQuery(ms)\tSecureQuery(MB)\tDataUpload(ms)\tDataUpload(MB)");
            for (int n : nValues) {
                long tToken=0, tQuery=0, tUpload=0; long mToken=0, mQuery=0, mUpload=0;
                for (int it=0; it<iterations; it++) {
                    ThreeDPPCostSimulation sim = new ThreeDPPCostSimulation(n, d, lambda, obstacles, landmarksArr[0], theta, c, csvFile);
                    sim.runSimulation();
                    tUpload += sim.stageStats.dataUpload.timeNanos;
                    tToken += sim.stageStats.tokenGeneration.timeNanos; tQuery += sim.stageStats.secureQuery.timeNanos;
                    mUpload = Math.max(mUpload, sim.stageStats.dataUpload.peakBytes);
                    mToken = Math.max(mToken, sim.stageStats.tokenGeneration.peakBytes);
                    mQuery = Math.max(mQuery, sim.stageStats.secureQuery.peakBytes);
                }
                double avgUploadMs = tUpload/(double)iterations/1_000_000.0;
                double avgTokenMs = tToken/(double)iterations/1_000_000.0;
                double avgQueryMs = tQuery/(double)iterations/1_000_000.0;
                long peakUploadMB = mUpload/(1024*1024);
                long peakTokenMB = mToken/(1024*1024);
                long peakQueryMB = mQuery/(1024*1024);
                System.out.printf("%d\t%.3f\t\t%d\t\t%.3f\t\t%d\t\t%.3f\t\t%d\n", n, avgTokenMs, peakTokenMB, avgQueryMs, peakQueryMB, avgUploadMs, peakUploadMB);
                pw.printf("4,%d,%d,%d,%d,%d,%d,TokenGen,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgTokenMs, peakTokenMB);
                pw.printf("4,%d,%d,%d,%d,%d,%d,SecureQuery,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgQueryMs, peakQueryMB);
                pw.printf("4,%d,%d,%d,%d,%d,%d,DataUpload,%.3f,%d\n", n, d, lambda, theta, c, landmarksArr[0], avgUploadMs, peakUploadMB);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // ===== Tree parameter impact cases 5-14 =====
    public static void case_tree_vary_c(int landmarks, String csvFile) {
        System.out.println("\n================================================================================");
        System.out.println("Tree Impact: n=10000, d=5, λ=128, theta=100, landmark="+landmarks+", varying c (2..8)");
        System.out.println("================================================================================");
        int n = 10000, d = 20, lambda = 128, obstacles = 50, theta = 100;
        int[] cValues = {2,3,4,5,6,7,8};
        int iterations = 20;
        String out = "tree_theta100_landmark"+landmarks+"_vary_c.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(out))) {
            pw.println("case,n,d,lambda,theta,c,landmarks,stage,time_ms,peak_mb");
            System.out.println("c\tTokenGen(ms)\tTokenGen(MB)\tSecureQuery(ms)\tSecureQuery(MB)\tDataUpload(ms)\tDataUpload(MB)");
            for (int c : cValues) {
                long tToken=0, tQuery=0, tUpload=0; long mToken=0, mQuery=0, mUpload=0;
                for (int it=0; it<iterations; it++) {
                    ThreeDPPCostSimulation sim = new ThreeDPPCostSimulation(n, d, lambda, obstacles, landmarks, theta, c, csvFile);
                    sim.runSimulation();
                    tUpload += sim.stageStats.dataUpload.timeNanos;
                    tToken += sim.stageStats.tokenGeneration.timeNanos;
                    tQuery += sim.stageStats.secureQuery.timeNanos;
                    mUpload = Math.max(mUpload, sim.stageStats.dataUpload.peakBytes);
                    mToken = Math.max(mToken, sim.stageStats.tokenGeneration.peakBytes);
                    mQuery = Math.max(mQuery, sim.stageStats.secureQuery.peakBytes);
                }
                double avgUploadMs = tUpload/(double)iterations/1_000_000.0;
                double avgTokenMs = tToken/(double)iterations/1_000_000.0;
                double avgQueryMs = tQuery/(double)iterations/1_000_000.0;
                long peakUploadMB = mUpload/(1024*1024);
                long peakTokenMB = mToken/(1024*1024);
                long peakQueryMB = mQuery/(1024*1024);
                System.out.printf("%d\t%.3f\t\t%d\t\t%.3f\t\t%d\t\t%.3f\t\t%d\n", c, avgTokenMs, peakTokenMB, avgQueryMs, peakQueryMB, avgUploadMs, peakUploadMB);
                pw.printf("5,%d,%d,%d,%d,%d,%d,TokenGen,%.3f,%d\n", n, d, lambda, theta, c, landmarks, avgTokenMs, peakTokenMB);
                pw.printf("5,%d,%d,%d,%d,%d,%d,SecureQuery,%.3f,%d\n", n, d, lambda, theta, c, landmarks, avgQueryMs, peakQueryMB);
                pw.printf("5,%d,%d,%d,%d,%d,%d,DataUpload,%.3f,%d\n", n, d, lambda, theta, c, landmarks, avgUploadMs, peakUploadMB);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void case_tree_vary_theta_c_fixed(int landmarks, String csvFile) {
        System.out.println("\n================================================================================");
        System.out.println("Tree Impact: n=10000, d=5, λ=128, c=100, landmark="+landmarks+", varying theta (50..400)");
        System.out.println("================================================================================");
        int n = 10000, d = 20, lambda = 128, obstacles = 50, c = 100;
        int[] thetaValues = {50,100,150,200,250,300,350,400};
        int iterations = 20;
        String out = "tree_c100_landmark"+landmarks+"_vary_theta.csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(out))) {
            pw.println("case,n,d,lambda,theta,c,landmarks,stage,time_ms,peak_mb");
            System.out.println("theta\tTokenGen(ms)\tTokenGen(MB)\tSecureQuery(ms)\tSecureQuery(MB)\tDataUpload(ms)\tDataUpload(MB)");
            for (int theta : thetaValues) {
                long tToken=0, tQuery=0, tUpload=0; long mToken=0, mQuery=0, mUpload=0;
                for (int it=0; it<iterations; it++) {
                    ThreeDPPCostSimulation sim = new ThreeDPPCostSimulation(n, d, lambda, obstacles, landmarks, theta, c, csvFile);
                    sim.runSimulation();
                    tUpload += sim.stageStats.dataUpload.timeNanos;
                    tToken += sim.stageStats.tokenGeneration.timeNanos;
                    tQuery += sim.stageStats.secureQuery.timeNanos;
                    mUpload = Math.max(mUpload, sim.stageStats.dataUpload.peakBytes);
                    mToken = Math.max(mToken, sim.stageStats.tokenGeneration.peakBytes);
                    mQuery = Math.max(mQuery, sim.stageStats.secureQuery.peakBytes);
                }
                double avgUploadMs = tUpload/(double)iterations/1_000_000.0;
                double avgTokenMs = tToken/(double)iterations/1_000_000.0;
                double avgQueryMs = tQuery/(double)iterations/1_000_000.0;
                long peakUploadMB = mUpload/(1024*1024); long peakTokenMB = mToken/(1024*1024);
                long peakQueryMB = mQuery/(1024*1024);
                System.out.printf("%d\t%.3f\t\t%d\t\t%.3f\t\t%d\t\t%.3f\t\t%d\n", theta, avgTokenMs, peakTokenMB, avgQueryMs, peakQueryMB, avgUploadMs, peakUploadMB);
                pw.printf("10,%d,%d,%d,%d,%d,%d,TokenGen,%.3f,%d\n", n, d, lambda, theta, c, landmarks, avgTokenMs, peakTokenMB);
                pw.printf("10,%d,%d,%d,%d,%d,%d,SecureQuery,%.3f,%d\n", n, d, lambda, theta, c, landmarks, avgQueryMs, peakQueryMB);
                pw.printf("10,%d,%d,%d,%d,%d,%d,DataUpload,%.3f,%d\n", n, d, lambda, theta, c, landmarks, avgUploadMs, peakUploadMB);
            }
        } catch (IOException e) { e.printStackTrace(); }
    }

    public static void main(String[] args) {
        String inputFile = "Florida_90.csv";
        if (args.length > 0) { inputFile = args[0]; }

        System.out.println("================================================================================");
        System.out.println("3DPPilot Performance Evaluation - Stage-Level (time + peak memory)");
        System.out.println("Only stages: Data Upload, Token Generation, Secure Skyline Query (exclude System Init & Result Retrieval)");
        System.out.println("================================================================================");

        case_vary_d_lambda_128(inputFile);
        case_vary_d_lambda_256(inputFile);
        case_vary_n_lambda_128(inputFile);
        case_vary_n_lambda_256(inputFile);
        for (int lm = 1; lm <= 5; lm++) case_tree_vary_c(lm, inputFile);
        for (int lm = 1; lm <= 5; lm++) case_tree_vary_theta_c_fixed(lm, inputFile);

        System.out.println("\n================================================================================");
        System.out.println("Evaluation Complete");
        System.out.println("CSV files written to working directory.");
        System.out.println("================================================================================");
    }
}
