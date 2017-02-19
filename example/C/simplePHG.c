/**************************************************************
*  Basic example of using Zoltan to partition a hypergraph.
*
* We think a hypergraph as a matrix, where the hyperedges are
* the rows, and the vertices are the columns.  If (i,j) is
* non-zero, this indicates that vertex j is in hyperedge i.
*
* In some Zoltan documentation, the non-zeroes in hypergraph 
* matrices are called "pins".
*
***************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "zoltan.h"

#define VTXTYPE_DC 1
#define VTXTYPE_TASK 2

/* Name of file containing hypergraph to be partitioned */
static char *global_fname="hypergraph.txt";
static char *global_out_fname="HP_out_hypergraph.txt";
/* 0 in screen, 1 in file */
static int SHOW_RESULT_IN_FILE=0;

/* Structure to hold distributed hypergraph */

typedef struct{

           /* Zoltan will partition vertices, while minimizing edge cuts */

  int numMyVertices;  /* number of vertices that I own initially */
  int numVtx;         /* number of vertices */
  ZOLTAN_ID_TYPE *vtxGID;        /* global ID of these vertices */
  float *vtxWts;          /* the array of vertices' weight */ 
  int * vtxType;             /* the type of vertices */ 
  int numMyHEdges;    /* number of my hyperedges */
  int numEdge;         /* number of edges */
  int numAllNbors; /* number of vertices in my hyperedges */
  ZOLTAN_ID_TYPE *edgeGID;       /* global ID of each of my hyperedges */
  float *edgeWts;     /* the array of edges' weight */
  int *fileCopy;      /* the number of copied file, modified */
  int *nborIndex;     /* index into nborGID array of edge's vertices */
  ZOLTAN_ID_TYPE *nborGID;  /* Vertices of edge edgeGID[i] begin at nborGID[nborIndex[i]] */
} HGRAPH_DATA;

/* 4 application defined query functions.  If we were going to define
 * a weight for each hyperedge, we would need to define 2 more query functions.
 * !!! 
 */

static int get_number_of_vertices(void *data, int *ierr);
static void get_vertex_list(void *data, int sizeGID, int sizeLID,
            ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                  int wgt_dim, float *obj_wgts, int *ierr);
static void get_hypergraph_size(void *data, int *num_lists, int *num_nonzeroes,
                                int *format, int *ierr);
static void get_hypergraph(void *data, int sizeGID, int num_edges, int num_nonzeroes,
                           int format, ZOLTAN_ID_PTR edgeGID, int *vtxPtr,
                           ZOLTAN_ID_PTR vtxGID, int *ierr);
 // !!!
static void get_edge_wts(void *data, int num_gid_entries,int num_lid_entries, int num_edges, int edge_weight_dim, ZOLTAN_ID_PTR edge_GID,
ZOLTAN_ID_PTR edge_LID, float  *edge_weight, int *ierr); 
static void get_edge_wts_size(void *data, int *num_edges, int *ierr);
/* Functions to read hypergraph in from file, distribute it, view it, handle errors */

static int get_next_line(FILE *fp, char *buf, int bufsize);
static int get_line_ints(char *buf, int bufsize, int *vals);
static void input_file_error(int numProcs, int tag, int startProc);
static void input_file_error_msg(int numProcs, int tag, int startProc,char * msg);
static void showHypergraph(int myProc, int numProcs, int *parts, HGRAPH_DATA *hg,char * toFile);
static void read_input_file(int myRank, int numProcs, char *fname, HGRAPH_DATA *data);

static HGRAPH_DATA global_hg; 

int main(int argc, char *argv[])
{
  int i, rc;
  float ver;
  struct Zoltan_Struct *zz;
  int changes, numGidEntries, numLidEntries, numImport, numExport;
  int myRank, numProcs;
  ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
  int *importProcs, *importToPart, *exportProcs, *exportToPart;
  int *parts;
  FILE *fp;
  HGRAPH_DATA hg;


  if(sizeof(argv)>2&&argv[1]!=NULL){
    global_fname = malloc(sizeof(char) * 100);
    strcpy(global_fname,argv[1]);

    global_out_fname = malloc(sizeof(char) * 100);
    strcpy(global_out_fname,global_fname);
    strcat(global_out_fname,".out");
  }
  if(sizeof(argv)>3&&argv[2]!=NULL){
    SHOW_RESULT_IN_FILE = atoi(argv[2]);
  }
  

  /******************************************************************
  ** Initialize MPI and Zoltan
  ******************************************************************/
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  rc = Zoltan_Initialize(argc, argv, &ver);

  if (rc != ZOLTAN_OK){
    printf("sorry...\n");
    MPI_Finalize();
    exit(0);
  }
  if(myRank==0){
    printf("Hello hypergraph partition world. --- from Proc%d\n", myRank);
  }
  /******************************************************************
  ** Read hypergraph from input file and distribute it 
  ******************************************************************/

  fp = fopen(global_fname, "r");
  if (!fp){
    if (myRank == 0) fprintf(stderr,"ERROR: Can not open %s\n",global_fname);
    MPI_Finalize();
    exit(1);
  }
  fclose(fp);

  read_input_file(myRank, numProcs, global_fname, &hg);

  // !!! 权值
  int *fileCopy = (int *) malloc(sizeof(int) * hg.numMyHEdges);
  memcpy(fileCopy, hg.fileCopy , sizeof(int) * hg.numMyHEdges);

  /******************************************************************
  ** Create a Zoltan library structure for this instance of load
  ** balancing.  Set the parameters and query functions that will
  ** govern the library's calculation.  See the Zoltan User's
  ** Guide for the definition of these and many other parameters.
  ******************************************************************/

  zz = Zoltan_Create(MPI_COMM_WORLD);

  /* General parameters */

  Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0"); /*debug level 0 to 10*/
  Zoltan_Set_Param(zz, "LB_METHOD", "HYPERGRAPH");   /* partitioning method */
  Zoltan_Set_Param(zz, "HYPERGRAPH_PACKAGE", "PHG"); /* version of method */
  Zoltan_Set_Param(zz, "NUM_GID_ENTRIES", "1");/* global IDs are integers */
  Zoltan_Set_Param(zz, "NUM_LID_ENTRIES", "1");/* local IDs are integers */
  Zoltan_Set_Param(zz, "RETURN_LISTS", "ALL"); /* export AND import lists */
  // !!! 权值 应该设置为1
  Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", "1"); /* use Zoltan default vertex weights */
  Zoltan_Set_Param(zz, "EDGE_WEIGHT_DIM", "1");/* use Zoltan default hyperedge weights */

  /* PHG parameters  - see the Zoltan User's Guide for many more
   *   (The "REPARTITION" approach asks Zoltan to create a partitioning that is
   *    better but is not too far from the current partitioning, rather than partitioning 
   *    from scratch.  It may be faster but of lower quality that LB_APPROACH=PARTITION.)
  */

  Zoltan_Set_Param(zz, "LB_APPROACH", "REPARTITION");

  /* Application defined query functions */

  Zoltan_Set_Num_Obj_Fn(zz, get_number_of_vertices, &hg);
  Zoltan_Set_Obj_List_Fn(zz, get_vertex_list, &hg);
  Zoltan_Set_HG_Size_CS_Fn(zz, get_hypergraph_size, &hg);
  //set fileCopy to hyperedge to zz
  Zoltan_Set_HG_CS_Custom_Fn(zz, get_hypergraph, &hg,fileCopy);
  Zoltan_Set_HG_Edge_Wts_Fn(zz, get_edge_wts, &hg);
  Zoltan_Set_HG_Size_Edge_Wts_Fn(zz, get_edge_wts_size, &hg);
  
  
  /******************************************************************
  ** Zoltan can now partition the vertices of hypergraph.
  ** In this simple example, we assume the number of partitions is
  ** equal to the number of processes.  Process rank 0 will own
  ** partition 0, process rank 1 will own partition 1, and so on.
  ******************************************************************/
  printf("Proc %d is ready... \n",myRank);
  rc = Zoltan_LB_Partition(zz, /* input (all remaining fields are output) */
        &changes,        /* 1 if partitioning was changed, 0 otherwise */ 
        &numGidEntries,  /* Number of integers used for a global ID */
        &numLidEntries,  /* Number of integers used for a local ID */
        &numImport,      /* Number of vertices to be sent to me */
        &importGlobalGids,  /* Global IDs of vertices to be sent to me */
        &importLocalGids,   /* Local IDs of vertices to be sent to me */
        &importProcs,    /* Process rank for source of each incoming vertex */
        &importToPart,   /* New partition for each incoming vertex */
        &numExport,      /* Number of vertices I must send to other processes*/
        &exportGlobalGids,  /* Global IDs of the vertices I must send */
        &exportLocalGids,   /* Local IDs of the vertices I must send */
        &exportProcs,    /* Process to which I send each of the vertices */
        &exportToPart);  /* Partition to which each vertex will belong */

  if (rc != ZOLTAN_OK){
    printf("sorry..., Proc %d is down.\n",myRank);
    MPI_Finalize();
    Zoltan_Destroy(&zz);
    exit(0);
  }
  /******************************************************************
  ** Visualize the hypergraph partitioning before and after calling Zoltan.
  ******************************************************************/

  parts = (int *)malloc(sizeof(int) * hg.numMyVertices);

  for (i=0; i < hg.numMyVertices; i++){
    parts[i] = myRank;
  }
  /*
  if (myRank== 0&&!SHOW_RESULT_IN_FILE){
    printf("\nHypergraph partition before calling Zoltan\n");
  }

  showHypergraph(myRank, numProcs, parts,&hg,NULL);
  */
  for (i=0; i < numExport; i++){
    parts[exportLocalGids[i]] = exportToPart[i];
  }
  
  if (myRank == 0&&!SHOW_RESULT_IN_FILE){
    printf("Graph partition after calling Zoltan\n");
  }

  showHypergraph(myRank, numProcs, parts,&hg,global_out_fname);

  /******************************************************************
  ** Free the arrays allocated by Zoltan_LB_Partition, and free
  ** the storage allocated for the Zoltan structure.
  ******************************************************************/

  Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, 
                      &importProcs, &importToPart);
  Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, 
                      &exportProcs, &exportToPart);

  Zoltan_Destroy(&zz);
  /**********************
  ** all done ***********
  **********************/
  printf("Proc %d says: all done, exiting... \n",myRank);
  MPI_Finalize();

  if (hg.numMyVertices > 0){
    free(parts);
    free(hg.vtxGID);
  }
  if (hg.numMyHEdges > 0){
    free(hg.edgeGID);
    free(hg.nborIndex);
    if (hg.numAllNbors > 0){
      free(hg.nborGID);
    }
  }

  return 0;
}

/* Application defined query functions */

static int get_number_of_vertices(void *data, int *ierr)
{
  HGRAPH_DATA *hg = (HGRAPH_DATA *)data;
  *ierr = ZOLTAN_OK;
  return hg->numMyVertices;
}

static void get_vertex_list(void *data, int sizeGID, int sizeLID,
            ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                  int wgt_dim, float *obj_wgts, int *ierr)
{
  // float *obj_wgts 设置权值
  int i;

  HGRAPH_DATA *hg= (HGRAPH_DATA *)data;
  *ierr = ZOLTAN_OK;

  /* In this example, return the IDs of our vertices, but no weights.
   * Zoltan will assume equally weighted vertices.
   */
   // !!! 设置节点权值
  for (i=0; i<hg->numMyVertices; i++){
    globalID[i] = hg->vtxGID[i];
    localID[i] = i;
    obj_wgts[i] = hg->vtxWts[i];
  }
}

static void get_hypergraph_size(void *data, int *num_lists, int *num_nonzeroes,
                                int *format, int *ierr)
{
  HGRAPH_DATA *hg = (HGRAPH_DATA *)data;
  *ierr = ZOLTAN_OK;

  *num_lists = hg->numMyHEdges;
  *num_nonzeroes = hg->numAllNbors;

  /* We will provide compressed hyperedge (row) format.  The alternative is
   * is compressed vertex (column) format: ZOLTAN_COMPRESSED_VERTEX.
   */

  *format = ZOLTAN_COMPRESSED_EDGE;

  return;
}

static void get_hypergraph(void *data, int sizeGID, int num_edges, int num_nonzeroes,
                           int format, ZOLTAN_ID_PTR edgeGID, int *vtxPtr,
                           ZOLTAN_ID_PTR vtxGID, int *ierr)
{
  int i;

  HGRAPH_DATA *hg = (HGRAPH_DATA *)data;
  *ierr = ZOLTAN_OK;
  // 验证是否正确
  if ( (num_edges != hg->numMyHEdges) || (num_nonzeroes != hg->numAllNbors) ||
       (format != ZOLTAN_COMPRESSED_EDGE)) {
    *ierr = ZOLTAN_FATAL;
    return;
  }

  for (i=0; i < num_edges; i++){
    edgeGID[i] = hg->edgeGID[i];
    vtxPtr[i] = hg->nborIndex[i];
  }

  for (i=0; i < num_nonzeroes; i++){
    vtxGID[i] = hg->nborGID[i];
  }

  return;
}

static void get_edge_wts(void *data, int num_gid_entries,int num_lid_entries,
   int num_edges, int edge_weight_dim, ZOLTAN_ID_PTR edge_GID,
                ZOLTAN_ID_PTR edge_LID, float  *edge_weight, int *ierr)
{
  
  HGRAPH_DATA *hg = (HGRAPH_DATA *)data;
  *ierr = ZOLTAN_OK;
  int i;
  if(edge_weight_dim==1){
    for (i=0; i < num_edges; i++){
      edge_GID[i] = hg->edgeGID[i];
      edge_LID[i] = i;
      edge_weight[i] = hg->edgeWts[i];
      //!!!
      //printf("%.1f, ",edge_weight[i]);
    }
  }else{
    *ierr = ZOLTAN_FATAL;
  }
  return;
}
static void get_edge_wts_size(void *data, int *num_edges, int *ierr){
   HGRAPH_DATA *hg = (HGRAPH_DATA *)data;
   *num_edges = hg->numMyHEdges;
   *ierr = ZOLTAN_OK;
}

/* Function to find next line of information in input file */
 
static int get_next_line(FILE *fp, char *buf, int bufsize)
{
int i, cval, len;
char *c;

  while (1){

    c = fgets(buf, bufsize, fp);

    if (c == NULL)
      return 0;  /* end of file */

    len = strlen(c);

    for (i=0, c=buf; i < len; i++, c++){
      cval = (int)*c; 
      if (isspace(cval) == 0) break;
    }
    if (i == len) continue;   /* blank line */
    if (*c == '#') continue;  /* comment */

    if (c != buf){
      strcpy(buf, c);
    }
    break;
  }

  return strlen(buf);  /* number of characters */
}

/* Function to return the list of non-negative integers in a line */

static int get_line_ints(char *buf, int bufsize, int *vals)
{
char *c = buf;
int count=0;

  while (1){
    while (!(isdigit(*c))){
      if ((c - buf) >= bufsize) break;
      c++;
    }
  
    if ( (c-buf) >= bufsize) break;
  
    vals[count++] = atoi(c);
  
    while (isdigit(*c)){
      if ((c - buf) >= bufsize) break;
      c++;
    }
  
    if ( (c-buf) >= bufsize) break;
  }

  return count;
}


/* Proc 0 notifies others of error and exits */
static void input_file_error_msg(int numProcs, int tag, int startProc,char * msg){
  int i, val[3];
  val[0] = -1;   /* error flag */

  fprintf(stderr,"ERROR in input file.\n");
  fprintf(stderr,"Detail:tag:%d numProcs:%d startProc:%d ,%s \n",tag,numProcs,startProc,msg);

  for (i=startProc; i < numProcs; i++){
    /* these procs have posted a receive for "tag" expecting counts */
    MPI_Send(val, 3, MPI_INT, i, tag, MPI_COMM_WORLD);
  }
  for (i=1; i < startProc; i++){
    /* these procs are done and waiting for ok-to-go */
    MPI_Send(val, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  exit(1);
}
static void input_file_error(int numProcs, int tag, int startProc)
{
  int i, val[3];
  val[0] = -1;   /* error flag */

  fprintf(stderr,"ERROR in input file.\n");
  fprintf(stderr,"Detail:tag:%d numProcs:%d startProc:%d\n",tag,numProcs,startProc);
  for (i=startProc; i < numProcs; i++){
    /* these procs have posted a receive for "tag" expecting counts */
    MPI_Send(val, 3, MPI_INT, i, tag, MPI_COMM_WORLD);
  }
  for (i=1; i < startProc; i++){
    /* these procs are done and waiting for ok-to-go */
    MPI_Send(val, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  exit(1);
}

/* Draw the partition assignments of the objects */

static void showHypergraph(int myProc, int numProcs, int *parts, HGRAPH_DATA *hg,char * toFile)
{
int numIDs = hg->numMyVertices;
ZOLTAN_ID_TYPE *GIDs = hg->vtxGID; 
int *partAssign = NULL, *allPartAssign = NULL;
int i, j, part, count, numVtx, numEdges;
int edgeIdx, vtxIdx;
int maxPart, nPart, partIdx;
int **M = NULL;
int *partNums = NULL, *partCount = NULL;
ZOLTAN_ID_TYPE *nextID;
ZOLTAN_ID_TYPE edgeID, vtxID;
int cutn, cutl;
int cutsize = 0;
float imbal, localImbal;

  numVtx = hg->numVtx;
  numEdges = hg->numEdge;

  partAssign = (int *)calloc(sizeof(int), numVtx);
  allPartAssign = (int *)calloc(sizeof(int), numVtx);

  for (i=0; i < numIDs; i++){
    partAssign[GIDs[i]-1] = parts[i];
  }

  MPI_Reduce(partAssign, allPartAssign, numVtx, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  free(partAssign);

  // 只在主线程中执行
  if (myProc > 0){
    free(allPartAssign);
    return;
  }

  /* Creating a dense matrix containing hyperedges, because this is small
   * example problem, and it is simpler.
   */

  M = (int **)calloc(sizeof(int *) , numEdges);
  for (i=0; i < numEdges; i++){
    M[i] = (int *)calloc(sizeof(int) , numVtx);
  }

  nextID = global_hg.nborGID;

  maxPart = 0;

  for (i=0; i < numEdges; i++){

    edgeID = global_hg.edgeGID[i];
    edgeIdx = (int)edgeID - 1;
    count = global_hg.nborIndex[i+1] - global_hg.nborIndex[i];

    for (j=0; j < count; j++){
      vtxID = *nextID++;
      vtxIdx = (int)vtxID - 1;

      part = allPartAssign[vtxIdx];

      if (part > maxPart) maxPart = part;

      M[edgeIdx][vtxIdx] = part+1;
    }
  }

  /* Calculate vertex balance measure 1.0 is perfect, higher is worse */

  imbal = 0;
  partCount = (int *)malloc(sizeof(int)* numProcs);
  for (i=0; i < numVtx; i++){
    partCount[allPartAssign[i]]++;
  }

  imbal = 0.0;
  for (part=0; part <= maxPart; part++){
     localImbal = (float)(numProcs * partCount[part]) / (float)numVtx;
     if (localImbal > imbal) imbal = localImbal;
  }

  if(SHOW_RESULT_IN_FILE){
    if(toFile!=NULL){
        /* Print the hypergraph to file */
        FILE *wfp;
        wfp=fopen(toFile,"w");
        if(wfp!=NULL)
        {
          int outParts = numProcs;
          int ** vtxsInPart = (int **)calloc(sizeof(int*), outParts);
          
          for(i=0;i<outParts;i++){
            vtxsInPart[i] = (int *)calloc(sizeof(int), numVtx+1);
            vtxsInPart[i][0] = 0;
          }
          
          for(i=0;i<numVtx;i++){
            int row = allPartAssign[i];
            vtxsInPart[row][vtxsInPart[row][0]+1] = i+1;
            vtxsInPart[row][0]++;
          }
          /* output total parts num */
          fprintf(wfp,"%d\n",outParts);
          for(i=0;i<outParts;i++){
            /* output each parts num */
            fprintf(wfp,"%d ",vtxsInPart[i][0]);
            /* output each parts vtx */
            for (j=0; j < vtxsInPart[i][0]; j++){
              fprintf(wfp,"%d ",vtxsInPart[i][j+1]);
            }
            fprintf(wfp,"\n");
          }

          for (i=0; i < outParts; i++){
            free(vtxsInPart[i]);
          }
          free(vtxsInPart);
          fclose(wfp);
        } else {
          fprintf(stderr,"ERROR in output file.\n");
        }
    }else{}
    
    
  }else{
    /* Print the hypergraph as a matrix */
    printf("\n                VERTICES\n    ");
    for (j=0; j < numVtx; j++){

      if (j < 9)
        printf("%d  ",j+1);
      else
        printf("%d ",j+1);
    }

    printf(" NPARTS-1");

    printf("\n    ");
    for (j=0; j < numVtx; j++){
      printf("---");
    }
    printf("\n");

    partNums = (int *)calloc(sizeof(int), maxPart + 1);
    cutn = 0;
    cutl = 0;

    for (i=0; i < numEdges; i++){
      nPart = 0;

      if (i < 9)
        printf("%d   ",i+1);
      else
        printf("%d  ",i+1);

      for (j=0; j < numVtx; j++){
        part = M[i][j];
        partIdx = part - 1;

        if (part > 0){
          printf("%d  ",partIdx);
          if (partNums[partIdx] < i+1){
            nPart++;
            partNums[partIdx] = i+1;
          }
        }
        else{
          printf("   ");
        }

      }
      if (nPart >= 2){
        printf("    %d\n",nPart - 1);
        cutn++;
        cutl += (nPart - 1);
      }
      else{
        printf("\n");
      }
    }
    printf("Total number of cut edges: %d\n",cutn);
    printf("Sum of NPARTS-1:           %d\n",cutl);
    printf("Balance of vertices across partitions: %f\n",imbal);
    printf("\n");

    free(partNums);
  }


  free(partCount);
  free(allPartAssign);
  for (i=0; i < numEdges; i++){
    free(M[i]);
  }
  free(M);
  
}

/*
 * Read the hypergraph in the input file and distribute the non-zeroes. (See the
 * matrix analogy at the top of the source file.)
 *
 * We will distribute the hyperedges (rows) to the processes.  However, we could
 * distribute the vertices (columns), or we could distribute the non-zeroes
 * instead.
 *
 * Zoltan partitions the vertices, so we also create an initial partitioning of the vertices.
 */

void read_input_file(int myRank, int numProcs, char *fname, HGRAPH_DATA *hg)
{
char buf[512];
int bufsize;
int numGlobalVertices, numGlobalEdges, numGlobalNZ;
int num, count, nnbors, ack=0;
int to=-1, from, remaining;
int vGID;
float vWts;
int vType;
int i, j;
int vals[128], send_count[5];
ZOLTAN_ID_TYPE *idx;
unsigned int id;
FILE *fp;
MPI_Status status;
int ack_tag = 5, count_tag = 10, id_tag = 15;
HGRAPH_DATA *send_hg;

  if (myRank == 0){

    bufsize = 512;

    fp = fopen(fname, "r");

    /* Get the number of vertices */

    num = get_next_line(fp, buf, bufsize);
    
    if (num == 0) input_file_error(numProcs,  count_tag, 1);
    num = sscanf(buf, "%d", &numGlobalVertices);
    if (num != 1) input_file_error(numProcs,  count_tag, 1);

    global_hg.numMyVertices = numGlobalVertices;
    global_hg.numVtx = numGlobalVertices;
    global_hg.vtxGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * numGlobalVertices);
    global_hg.vtxWts = (float *) malloc(sizeof(float) * numGlobalVertices);
    global_hg.vtxType = (int *) malloc(sizeof(int) * numGlobalVertices);
    /* Get the vertex global IDs and its according weight */

    for (i=0; i < numGlobalVertices; i++){

      num = get_next_line(fp, buf, bufsize);
      if (num == 0) input_file_error(numProcs,  count_tag, 1);
      num = sscanf(buf, "%d %f %d", &vGID,&vWts,&vType);
      if (num != 3) input_file_error(numProcs,  count_tag, 1);

      global_hg.vtxGID[i] = (ZOLTAN_ID_TYPE)vGID;
      global_hg.vtxWts[i] = vWts;
      global_hg.vtxType[i] = vType;
    }

    /* Get the number hyperedges which contain those vertices */

    num = get_next_line(fp, buf, bufsize);
    if (num == 0) input_file_error(numProcs,  count_tag, 1);
    num = sscanf(buf, "%d", &numGlobalEdges);
    if (num != 1) input_file_error(numProcs,  count_tag, 1);

    global_hg.numMyHEdges = numGlobalEdges;
    global_hg.numEdge = numGlobalEdges;

    global_hg.edgeGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * numGlobalEdges);
    global_hg.edgeWts = (float *) malloc(sizeof(float) * numGlobalEdges);
    global_hg.fileCopy = (int *)malloc(sizeof(int) * numGlobalEdges);
    global_hg.nborIndex = (int *)malloc(sizeof(int) * (numGlobalEdges + 1));

    /* Get the total number of vertices or neighbors in all the hyperedges of
     * the hypergraph.  Or get the number of non-zeroes in the matrix representing
     * the hypergraph.
     */

    num = get_next_line(fp, buf, bufsize);
    if (num == 0) input_file_error(numProcs,  count_tag, 1);
    num = sscanf(buf, "%d", &numGlobalNZ);
    if (num != 1) input_file_error(numProcs,  count_tag, 1);

    global_hg.nborGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * numGlobalNZ);

    /* Get the list of vertices in each hyperedge  */

    global_hg.nborIndex[0] = 0;

    for (i=0; i < numGlobalEdges; i++){

      num = get_next_line(fp, buf, bufsize);
      if (num == 0) input_file_error(numProcs,  count_tag, 1);

      num = get_line_ints(buf, bufsize, vals);

      if (num < 2) input_file_error(numProcs,  count_tag, 1);

      id = vals[0];
      nnbors = vals[1];

      if (num < (nnbors + 2)) input_file_error(numProcs,  count_tag, 1);

      global_hg.edgeGID[i] = (ZOLTAN_ID_TYPE)id;

      for (j=0; j < nnbors; j++){
        global_hg.nborGID[global_hg.nborIndex[i] + j] = (ZOLTAN_ID_TYPE)vals[2 + j];
      }

      global_hg.nborIndex[i+1] = global_hg.nborIndex[i] + nnbors;
    }
    int eGID;
    float eWts;

    /* Get the weight of hyperedges */
    for (i=0; i < numGlobalEdges; i++){
      num = get_next_line(fp, buf, bufsize);
      if (num == 0) input_file_error(numProcs,  count_tag, 1);
      
       num = sscanf(buf, "%d %f", &eGID,&eWts);
      if (num != 2) input_file_error(numProcs,  count_tag, 1);
      
      global_hg.edgeWts[i] = eWts;
    } 
    fclose(fp);

    /* calcuate the fileCopy of each hyperedge */
    for(i=0;i<numGlobalEdges;i++){
      int count = 0;
      int nnbors = 0;
      nnbors = global_hg.nborIndex[i+1] - global_hg.nborIndex[i];
      for (j=0; j < nnbors; j++){
        ZOLTAN_ID_TYPE vtxid = global_hg.nborGID[global_hg.nborIndex[i] + j];
        int k;
        for(k=0;k<numGlobalVertices;k++){
          //find the vtx
          if(global_hg.vtxGID[k]==vtxid){
            if(global_hg.vtxType[k]==VTXTYPE_DC){
              //find file(hyperedge) has a copy in dc k
              count++;
            }
            break;
          }
        }
      }
      if(count<1){
        input_file_error_msg(numProcs,  count_tag, 1, "fileCopy error, must bigger than 1.");
      }
      else{
        global_hg.fileCopy[i] = count;
      }
    }

    /* Create a sub graph for each process */

    send_hg = (HGRAPH_DATA *)calloc(sizeof(HGRAPH_DATA) , numProcs);

    /* 
     * Divide the vertices across the processes
     */

    remaining = numGlobalVertices;
    count = (numGlobalVertices / numProcs) + 1;
    idx = global_hg.vtxGID;

    for (i=0; i < numProcs; i++){

      if (remaining == 0) count = 0;
      if (count > remaining) count = remaining;

      send_hg[i].numMyVertices = count;

      if (count){

        send_hg[i].vtxGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * count);
        //vertexs weight copy
        send_hg[i].vtxWts = (float *)malloc(sizeof(float) * count);
        send_hg[i].vtxType = (int *)malloc(sizeof(int) * count);
        for (j=0; j < count; j++){
          send_hg[i].vtxWts[j] = global_hg.vtxWts[*idx];
          send_hg[i].vtxType[j] = global_hg.vtxType[*idx];
          send_hg[i].vtxGID[j] = *idx++;
        }
      }

      remaining -= count;
    }

    /*
     * Assign hyperedges to processes, and create a sub-hypergraph for each process.
     */

    remaining = numGlobalEdges;
    count = (numGlobalEdges / numProcs) + 1;
    from = 0;

    for (i=0; i < numProcs; i++){

      if (remaining == 0) count = 0;
      if (count > remaining) count = remaining;

      send_hg[i].numMyHEdges = count;
      send_hg[i].numAllNbors = 0;

      if (count > 0){
  
        to = from + count;
  
        nnbors = global_hg.nborIndex[to] - global_hg.nborIndex[from];

        send_hg[i].numAllNbors = nnbors;
      
        send_hg[i].edgeGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * count);
        memcpy(send_hg[i].edgeGID, global_hg.edgeGID + from, sizeof(ZOLTAN_ID_TYPE) * count);
        //edge weight copy
        send_hg[i].edgeWts = (float *) malloc(sizeof(float) * count);
        memcpy(send_hg[i].edgeWts, global_hg.edgeWts + from, sizeof(float) * count);
        //fileCopy copy
        send_hg[i].fileCopy = (int *) malloc(sizeof(int) * count);
        memcpy(send_hg[i].fileCopy, global_hg.fileCopy + from, sizeof(int) * count);

        send_hg[i].nborIndex = (int *)malloc(sizeof(int) * (count + 1));
        send_hg[i].nborIndex[0] = 0;

        if (nnbors > 0){

          num = global_hg.nborIndex[from];

          for (j=1; j <= count; j++){
            send_hg[i].nborIndex[j] = global_hg.nborIndex[from+j] - num;
          }
        
          send_hg[i].nborGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * nnbors);
          memcpy(send_hg[i].nborGID, 
                 global_hg.nborGID + global_hg.nborIndex[from], 
                 sizeof(ZOLTAN_ID_TYPE) * nnbors);
        }
      }

      remaining -= count;
      from = to;
    }

    /* Send each process its hyperedges and the vertices in its partition */
    // 主线程 0 任务
    *hg = send_hg[0];
    hg->numVtx = global_hg.numVtx;
    hg->numEdge = global_hg.numEdge;

    for (i=1; i < numProcs; i++){
      send_count[0] = send_hg[i].numMyVertices;
      send_count[1] = send_hg[i].numMyHEdges;
      send_count[2] = send_hg[i].numAllNbors;
      send_count[3] = global_hg.numVtx;
      send_count[4] = global_hg.numEdge;

      MPI_Send(send_count, 5, MPI_INT, i, count_tag, MPI_COMM_WORLD);
      MPI_Recv(&ack, 1, MPI_INT, i, ack_tag, MPI_COMM_WORLD, &status);

      if (send_count[0] > 0){
        MPI_Send(send_hg[i].vtxGID, send_count[0], ZOLTAN_ID_MPI_TYPE, i, id_tag, MPI_COMM_WORLD);
        MPI_Send(send_hg[i].vtxWts, send_count[0], MPI_FLOAT, i, id_tag+1, MPI_COMM_WORLD);
        MPI_Send(send_hg[i].vtxType, send_count[0], MPI_INT, i, id_tag+2, MPI_COMM_WORLD);
        free(send_hg[i].vtxGID);
        free(send_hg[i].vtxWts);
      }

      if (send_count[1] > 0){
        MPI_Send(send_hg[i].edgeGID, send_count[1], ZOLTAN_ID_MPI_TYPE, i, id_tag + 3, MPI_COMM_WORLD);
        MPI_Send(send_hg[i].edgeWts, send_count[1], MPI_FLOAT, i, id_tag + 4, MPI_COMM_WORLD);
        MPI_Send(send_hg[i].fileCopy, send_count[1], MPI_INT, i, id_tag + 5, MPI_COMM_WORLD);
        free(send_hg[i].edgeGID);
        free(send_hg[i].edgeWts);
        free(send_hg[i].fileCopy);

        MPI_Send(send_hg[i].nborIndex, send_count[1] + 1, MPI_INT, i, id_tag + 6, MPI_COMM_WORLD);
        free(send_hg[i].nborIndex);

        if (send_count[2] > 0){
          MPI_Send(send_hg[i].nborGID, send_count[2], ZOLTAN_ID_MPI_TYPE, i, id_tag + 7, MPI_COMM_WORLD);
          free(send_hg[i].nborGID);
        }
      }
    }

    free(send_hg);

    /* signal all procs it is OK to go on */
    ack = 0;
    for (i=1; i < numProcs; i++){
      MPI_Send(&ack, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  }
  else{// if(myRank==0)

    MPI_Recv(send_count, 5, MPI_INT, 0, count_tag, MPI_COMM_WORLD, &status);

    if (send_count[0] < 0){
      MPI_Finalize();
      exit(1);
    }
    

    ack = 0;

    memset(hg, 0, sizeof(HGRAPH_DATA));

    hg->numMyVertices = send_count[0];
    hg->numMyHEdges = send_count[1];
    hg->numAllNbors = send_count[2];
    hg->numVtx = send_count[3];
    hg->numEdge = send_count[4];

    if (send_count[0] > 0){
      hg->vtxGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * send_count[0]);
      hg->vtxWts = (float *)malloc(sizeof(float) * send_count[0]);
      hg->vtxType = (int *)malloc(sizeof(int) * send_count[0]);
    }

    if (send_count[1] > 0){
      hg->edgeGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * send_count[1]);
      hg->edgeWts = (float *)malloc(sizeof(float) * send_count[1]);
      hg->fileCopy = (int *)malloc(sizeof(int) * send_count[1]);
      hg->nborIndex = (int *)malloc(sizeof(int) * (send_count[1] + 1));

      if (send_count[2] > 0){
        hg->nborGID = (ZOLTAN_ID_TYPE *)malloc(sizeof(ZOLTAN_ID_TYPE) * send_count[2]);
      }
    }
    MPI_Send(&ack, 1, MPI_INT, 0, ack_tag, MPI_COMM_WORLD);

    if (send_count[0] > 0){
      MPI_Recv(hg->vtxGID,send_count[0], ZOLTAN_ID_MPI_TYPE, 0, id_tag, MPI_COMM_WORLD, &status);
      MPI_Recv(hg->vtxWts,send_count[0], MPI_FLOAT, 0, id_tag+1, MPI_COMM_WORLD, &status);
      MPI_Recv(hg->vtxType,send_count[0], MPI_INT, 0, id_tag+2, MPI_COMM_WORLD, &status);

      if (send_count[1] > 0){
        MPI_Recv(hg->edgeGID,send_count[1], ZOLTAN_ID_MPI_TYPE, 0, id_tag + 3, MPI_COMM_WORLD, &status);
        MPI_Recv(hg->edgeWts,send_count[1], MPI_FLOAT, 0, id_tag + 4, MPI_COMM_WORLD, &status);
        MPI_Recv(hg->fileCopy,send_count[1], MPI_INT, 0, id_tag + 5, MPI_COMM_WORLD, &status);
        MPI_Recv(hg->nborIndex,send_count[1] + 1, MPI_INT, 0, id_tag + 6, MPI_COMM_WORLD, &status);

        if (send_count[2] > 0){
          MPI_Recv(hg->nborGID,send_count[2], ZOLTAN_ID_MPI_TYPE, 0, id_tag + 7, MPI_COMM_WORLD, &status);
        }
      }
    }

    /* ok to go on? */

    MPI_Recv(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    if (ack < 0){
      MPI_Finalize();
      exit(1);
    }
  }
}
