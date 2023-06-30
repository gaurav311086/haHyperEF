#ifndef HYPEREF_H_
#define HYPEREF_H_

#include <math.h>
#include <random>
#include <algorithm>
#include "data_loader.h"

// Data structure for csc matrix.
template<typename data_type>
struct SVEC {
    /*! \brief The number of rows of the sparse vector */
    uint32_t size;
    /*! \brief The non-zero data of the sparse vector */
    std::vector<data_type> adj_data;
    /*! \brief The index pointers of the sparse vector */
    std::vector<uint32_t> adj_indptr;
};

int get_random(int min, int max)
{
    /*static std::mt19937 mt{ std::random_device{}() };
    std::uniform_int_distribution die{ min, max };

    return die(mt);*/
	return rand();
}

// Load a csc matrix from a hmetis (unweighted hypergraph) file. The sparse matrix should have float data type.
spmv::io::CSCMatrix<float> load_csr_matrix_from_float_hmetis_unweighted(std::string csc_float_hmetis_path) {
    spmv::io::CSCMatrix<float> csc_matrix;
    std::ifstream io;
    io.open(csc_float_hmetis_path.c_str());
    if ( !io.is_open() ){
        throw std::runtime_error("FATAL: failed ifstream open");
    }
    std::string line;
    uint32_t lines_read;
    uint32_t this_hedge_node;
    uint32_t nnz_elements_read;
    nnz_elements_read = 0;
    lines_read=0;
    while (io.good()){
        bool first_int;
        bool second_int;
        line.clear();
        getline(io, line);
        std::string substr;
        substr.clear();
        first_int = false;
        second_int = false;
        for(uint32_t i=0; i < line.length();i++){
          if(line[i] == ' ') {
              if(substr.length() > 0){
                if(lines_read>0){
                    nnz_elements_read++;
                    this_hedge_node = std::stoi(substr.c_str());
                  csc_matrix.adj_indices.push_back(this_hedge_node-1);
                  csc_matrix.adj_data.push_back(1.0);
                } else {
                  if(first_int){
                      if(second_int){
                          io.close();
                          throw std::runtime_error("FATAL: Only unweighted hmetis files supported");
                      }
                      csc_matrix.num_rows = std::stoi(substr.c_str());
                      second_int=true;
                  } else {
                    csc_matrix.num_cols = std::stoi(substr.c_str());
                    first_int = true;
                  }
                }
              }
              substr.clear();
          } else {
              if(!(line[i]=='\r' || line[i]=='\n')) {
                      substr.push_back(line[i]);
              } else {
                substr.clear();
              }
          }
        }
        if(line.length()>0) {
            csc_matrix.adj_indptr.push_back(nnz_elements_read);
        }
        lines_read++;
    }
    io.close();
    return csc_matrix;
}

// Convert csc to csr.
template<typename data_type>
spmv::io::CSRMatrix<data_type> csc2csr(spmv::io::CSCMatrix<data_type> const &csc_matrix) {
    spmv::io::CSRMatrix<data_type> csr_matrix;
    csr_matrix.num_rows = csc_matrix.num_rows;
    csr_matrix.num_cols = csc_matrix.num_cols;
    csr_matrix.adj_data = std::vector<data_type>(csc_matrix.adj_data.size());
    csr_matrix.adj_indices = std::vector<uint32_t>(csc_matrix.adj_indices.size());
    csr_matrix.adj_indptr = std::vector<uint32_t>(csr_matrix.num_rows + 1);
    // Convert adj_indptr
    uint32_t nnz = csc_matrix.adj_indptr[csc_matrix.num_cols];
    std::vector<uint32_t> nnz_each_row(csr_matrix.num_rows);
    std::fill(nnz_each_row.begin(), nnz_each_row.end(), 0);
    for (size_t n = 0; n < nnz; n++) {
        nnz_each_row[csc_matrix.adj_indices[n]]++;
    }
    csr_matrix.adj_indptr[0] = 0;
    for (size_t row_idx = 0; row_idx < csr_matrix.num_rows; row_idx++) {
        csr_matrix.adj_indptr[row_idx + 1] = csr_matrix.adj_indptr[row_idx] + nnz_each_row[row_idx];
    }
    assert(csr_matrix.adj_indptr[csr_matrix.num_rows] == nnz);
    // Convert adj_data and adj_indices
    std::vector<uint32_t> nnz_consumed_each_row(csr_matrix.num_rows);
    std::fill(nnz_consumed_each_row.begin(), nnz_consumed_each_row.end(), 0);
    for (size_t col_idx = 0; col_idx < csc_matrix.num_cols; col_idx++){
        for (size_t i = csc_matrix.adj_indptr[col_idx]; i < csc_matrix.adj_indptr[col_idx + 1]; i++){
            uint32_t row_idx = csc_matrix.adj_indices[i];
            uint32_t dest = csr_matrix.adj_indptr[row_idx] + nnz_consumed_each_row[row_idx];
            csr_matrix.adj_indices[dest] = col_idx;
            csr_matrix.adj_data[dest] = csc_matrix.adj_data[i];
            nnz_consumed_each_row[row_idx]++;
        }
    }
    for (size_t row_idx = 0; row_idx < csr_matrix.num_rows; row_idx++) {
        assert(nnz_consumed_each_row[row_idx] == nnz_each_row[row_idx]);
    }
    return csr_matrix;
}

template<typename data_type>
spmv::io::CSCMatrix<data_type> transpose(spmv::io::CSCMatrix<data_type> const &csc_matrix) {
    spmv::io::CSCMatrix<data_type> csr_matrix;
    csr_matrix.num_rows = csc_matrix.num_cols;
    csr_matrix.num_cols = csc_matrix.num_rows;
    csr_matrix.adj_data = std::vector<data_type>(csc_matrix.adj_data.size());
    csr_matrix.adj_indices = std::vector<uint32_t>(csc_matrix.adj_indices.size());
    csr_matrix.adj_indptr = std::vector<uint32_t>(csr_matrix.num_rows + 1);
    // Convert adj_indptr
    uint32_t nnz = csc_matrix.adj_indptr[csc_matrix.num_cols];
    std::vector<uint32_t> nnz_each_row(csr_matrix.num_rows);
    std::fill(nnz_each_row.begin(), nnz_each_row.end(), 0);
    for (size_t n = 0; n < nnz; n++) {
        nnz_each_row[csc_matrix.adj_indices[n]]++;
    }
    csr_matrix.adj_indptr[0] = 0;
    for (size_t row_idx = 0; row_idx < csr_matrix.num_rows; row_idx++) {
        csr_matrix.adj_indptr[row_idx + 1] = csr_matrix.adj_indptr[row_idx] + nnz_each_row[row_idx];
    }
    assert(csr_matrix.adj_indptr[csr_matrix.num_rows] == nnz);
    // Convert adj_data and adj_indices
    std::vector<uint32_t> nnz_consumed_each_row(csr_matrix.num_rows);
    std::fill(nnz_consumed_each_row.begin(), nnz_consumed_each_row.end(), 0);
    for (size_t col_idx = 0; col_idx < csc_matrix.num_cols; col_idx++){
        for (size_t i = csc_matrix.adj_indptr[col_idx]; i < csc_matrix.adj_indptr[col_idx + 1]; i++){
            uint32_t row_idx = csc_matrix.adj_indices[i];
            uint32_t dest = csr_matrix.adj_indptr[row_idx] + nnz_consumed_each_row[row_idx];
            csr_matrix.adj_indices[dest] = col_idx;
            csr_matrix.adj_data[dest] = csc_matrix.adj_data[i];
            nnz_consumed_each_row[row_idx]++;
        }
    }
    for (size_t row_idx = 0; row_idx < csr_matrix.num_rows; row_idx++) {
        assert(nnz_consumed_each_row[row_idx] == nnz_each_row[row_idx]);
    }
    return csr_matrix;
}

template<typename data_type>
spmv::io::CSCMatrix<data_type> create_diagonal_matrix(std::vector<data_type> const &W, uint32_t rows_cols = 0) {
    spmv::io::CSCMatrix<data_type> csc_matrix;
    uint32_t diagonal_elements;
    uint32_t nWeights;
    if(rows_cols)
        diagonal_elements = rows_cols;
    else
        diagonal_elements = W.size();
    nWeights = W.size();
    csc_matrix.num_rows=diagonal_elements;
    csc_matrix.num_cols=diagonal_elements;
    csc_matrix.adj_data = std::vector<data_type>(diagonal_elements);
    csc_matrix.adj_indices = std::vector<uint32_t>(diagonal_elements);
    csc_matrix.adj_indptr = std::vector<uint32_t>(diagonal_elements+1);
    for(uint32_t i=0; i<diagonal_elements;i++){
        if(nWeights==1)
            csc_matrix.adj_data[i]=W[0];
        else
            csc_matrix.adj_data[i]=W[i];
        csc_matrix.adj_indices[i]=i;
        csc_matrix.adj_indptr[i]=i;
    }
    csc_matrix.adj_indptr[diagonal_elements]=diagonal_elements;
    return csc_matrix;
}

template<typename data_type>
spmv::io::CSCMatrix<data_type> sum(spmv::io::CSCMatrix<data_type> const &csc_matrix1, spmv::io::CSCMatrix<data_type> const &csc_matrix2){
    spmv::io::CSCMatrix<data_type> csc_matrixOut;
    uint32_t sz1;
    uint32_t sz2;
    uint32_t elemCovered1;
    uint32_t elemCovered2;
    uint32_t elemAdded;
    if(csc_matrix1.num_rows != csc_matrix2.num_rows)
        throw std::runtime_error("FATAL: Matrix dimensions do match..check rows");
    if(csc_matrix1.num_cols != csc_matrix2.num_cols)
        throw std::runtime_error("FATAL: Matrix dimensions do match..check columns");
    sz1 = csc_matrix1.adj_data.size();
    sz2 = csc_matrix2.adj_data.size();
    csc_matrixOut.num_rows=csc_matrix1.num_rows;
    csc_matrixOut.num_cols=csc_matrix2.num_cols;
    csc_matrixOut.adj_indptr = std::vector<uint32_t>(csc_matrixOut.num_cols + 1);
    elemCovered1 = 0;
    elemCovered2 = 0;
    elemAdded    = 0;

    csc_matrixOut.adj_indptr[0]=elemAdded;
    for(uint32_t col=0;col<csc_matrixOut.num_cols;col++){
        uint32_t nnz_ele_in_mat1;
        uint32_t nnz_ele_in_mat2;
        uint32_t ToBeTraversed;
        uint32_t nnz_traversed_this_step;
        nnz_ele_in_mat1 = csc_matrix1.adj_indptr[col+1]-csc_matrix1.adj_indptr[col];
        nnz_ele_in_mat2 = csc_matrix2.adj_indptr[col+1]-csc_matrix2.adj_indptr[col];
        nnz_traversed_this_step=0;
        ToBeTraversed = nnz_ele_in_mat1 + nnz_ele_in_mat2;
        for(uint32_t traversed=0; traversed < ToBeTraversed; traversed+=nnz_traversed_this_step){
            nnz_traversed_this_step=0;
            if((nnz_ele_in_mat1 == 0) && (nnz_ele_in_mat2==0)){
                nnz_traversed_this_step=0;
                //throw std::runtime_error("FATAL: Loop error");
            } else {
                if(nnz_ele_in_mat1 == 0) {
                    //matrix1 has no element in this row
                    //  but mat2 has some. copy those
                    uint32_t copy_ops;
                    copy_ops= nnz_ele_in_mat2;
                    for(uint32_t iter=0;iter<copy_ops;iter++){
                        csc_matrixOut.adj_data.push_back(csc_matrix2.adj_data[elemCovered2]);
                        csc_matrixOut.adj_indices.push_back(csc_matrix2.adj_indices[elemCovered2]);
                        elemCovered2++;
                        nnz_traversed_this_step++;
                        elemAdded++;
                        nnz_ele_in_mat2--;
                    }
                } else if(nnz_ele_in_mat2 == 0) {
                    //matrix2 has no element in this row
                    //  but mat1 has some. copy those
                    uint32_t copy_ops;
                    copy_ops= nnz_ele_in_mat1;
                    for(uint32_t iter=0;iter<copy_ops;iter++){
                        csc_matrixOut.adj_data.push_back(csc_matrix1.adj_data[elemCovered1]);
                        csc_matrixOut.adj_indices.push_back(csc_matrix1.adj_indices[elemCovered1]);
                        elemCovered1++;
                        nnz_traversed_this_step++;
                        elemAdded++;
                        nnz_ele_in_mat1--;
                    }
                } else {
                    //Both Matrix have some elements.
                    if(csc_matrix1.adj_indices[elemCovered1] == csc_matrix2.adj_indices[elemCovered2]){
                        // same row entry present..add
                        csc_matrixOut.adj_data.push_back(csc_matrix1.adj_data[elemCovered1]+csc_matrix2.adj_data[elemCovered2]);
                        csc_matrixOut.adj_indices.push_back(csc_matrix1.adj_indices[elemCovered1]);
                        elemCovered1++;
                        elemCovered2++;
                        nnz_traversed_this_step+=2;
                        nnz_ele_in_mat1--;
                        nnz_ele_in_mat2--;
                    } else if(csc_matrix1.adj_indices[elemCovered1] < csc_matrix2.adj_indices[elemCovered2]){
                        // append mat1 entry now
                        csc_matrixOut.adj_data.push_back(csc_matrix1.adj_data[elemCovered1]);
                        csc_matrixOut.adj_indices.push_back(csc_matrix1.adj_indices[elemCovered1]);
                        elemCovered1++;
                        nnz_traversed_this_step+=1;
                        nnz_ele_in_mat1--;
                    } else {
                        //append mat2 entry now
                        csc_matrixOut.adj_data.push_back(csc_matrix2.adj_data[elemCovered2]);
                        csc_matrixOut.adj_indices.push_back(csc_matrix2.adj_indices[elemCovered2]);
                        elemCovered2++;
                        nnz_traversed_this_step+=1;
                        nnz_ele_in_mat2--;
                    }
                    elemAdded++;
                }
            }
        }
        csc_matrixOut.adj_indptr[col+1]=elemAdded;
    }
    return csc_matrixOut;
}

spmv::io::CSCMatrix<float> StarW(spmv::io::CSCMatrix<float> const &ar, std::vector<float> &W){
    uint32_t mx;
    uint32_t sz;

    mx = ar.num_rows;
    sz = ar.num_cols;

    spmv::io::CSCMatrix <float> matrix;
    spmv::io::CSCMatrix <float> matrixT;
    spmv::io::CSCMatrix <float> matrixO;
    matrix.num_rows= mx + sz;
    matrix.num_cols= mx + sz;
    matrixT.num_rows= mx + sz;
    matrixT.num_cols= mx + sz;
    matrix.adj_data = std::vector<float>(ar.adj_data.size());
    matrix.adj_indices = std::vector<uint32_t>(ar.adj_indices.size());
    matrix.adj_indptr = std::vector<uint32_t>(mx+sz + 1);
    for(uint32_t i=0; i<mx+1;i++){
        matrix.adj_indptr[i] = 0;
    }
    uint32_t indices_covered;
    uint32_t lastArIndptr;
    indices_covered = 0;
    lastArIndptr = ar.adj_indptr[0];
    for(uint32_t iter =0;iter<sz;iter++){
        uint32_t LN;
        float edge_weight;
        uint32_t currArIndptr;
        currArIndptr = ar.adj_indptr[iter+1];
        matrix.adj_indptr[iter+mx+1] = currArIndptr;
        LN= currArIndptr - lastArIndptr;
        edge_weight=1.0/float(LN);
        for(uint32_t j=0; j< LN; j++){
            matrix.adj_data[indices_covered]=edge_weight;
            matrix.adj_indices[indices_covered]=ar.adj_indices[indices_covered];
            indices_covered++;
        }
        lastArIndptr=currArIndptr;
    }
    matrixT = transpose(matrix);
    matrixO = sum(matrix,matrixT);
    return matrixO;

}

spmv::io::CSCMatrix<float> SubRoutine001Filter(spmv::io::CSCMatrix <float> &AD){
    // = sparse(I2, I2, sparsevec(dg))
    spmv::io::CSCMatrix<float> csc_matrix;
    uint32_t ElementsSummed;
    uint32_t ElementsAdded;
    csc_matrix.num_rows=AD.num_rows;
    csc_matrix.num_cols=AD.num_cols;
    csc_matrix.adj_indptr=std::vector<uint32_t>(csc_matrix.num_cols+1);
    ElementsSummed=AD.adj_indptr[0];
    ElementsAdded=0;
    for(uint32_t col=0;col<AD.num_cols;col++){
        uint32_t ElementsToBeSummed;
        float sum;
        ElementsToBeSummed = AD.adj_indptr[col+1] - ElementsSummed;
        sum=0;
        for(uint32_t ele=0;ele<ElementsToBeSummed;ele++){
            sum+=AD.adj_data[ele+ElementsSummed];
        }
        ElementsSummed = AD.adj_indptr[col+1];
        if(ElementsToBeSummed){
            ElementsAdded++;
            sum = pow(sum, (-0.5));
            csc_matrix.adj_data.push_back(sum);
            csc_matrix.adj_indices.push_back(col);
        }
        csc_matrix.adj_indptr[col+1]=ElementsAdded;
    }
    return csc_matrix;
}

std::vector<float> SubRoutine002Filter(std::vector <float> &rv, uint32_t sz){
    std::vector<float> new_vector;
    //=rv - ((dot(rv, on) / dot(on, on)) * on)
    float dot_product;
    float mean_dot_product;

    new_vector=std::vector<float>(sz);
    dot_product=0;
    for(uint32_t i=0;i<sz;i++) {
        dot_product += rv[i];
    }
    mean_dot_product=dot_product/float(sz);
    for(uint32_t i=0;i<sz;i++){
        new_vector[i]=rv[i]-mean_dot_product;
    }
    return new_vector;
}



std::vector<float> SubRoutine003Filter(std::vector <float> &sm_ot,uint32_t sz,float power=2){
    std::vector<float> new_vector;
    //= sm_ot ./ norm(sm_ot)
    float dot_product;
    float norm_dot_product;

    new_vector=std::vector<float>(sz);
    dot_product=0;
    for(uint32_t i=0;i<sz;i++) {
        dot_product += pow(sm_ot[i],power);
    }
    norm_dot_product=pow(dot_product,1/power);
    for(uint32_t i=0;i<sz;i++){
        new_vector[i]=sm_ot[i]/norm_dot_product;
    }
    return new_vector;
}

std::vector<float> SubRoutine004Filter(spmv::io::CSCMatrix <float> &D, std::vector <float> &sm){
    //=D*sm
    std::vector<float> new_vector;
    uint32_t sz;
    sz=D.num_rows;
    new_vector=std::vector<float>(sz);
    std::fill(new_vector.begin(), new_vector.end(), 0);
    for (uint32_t row_idx = 0; row_idx < sz; row_idx++) {
        uint32_t start;
        uint32_t end;
        start = D.adj_indptr[row_idx];
        end = D.adj_indptr[row_idx + 1];
        for (uint32_t i = start; i < end; i++) {
            uint32_t idx;
            idx = D.adj_indices[i];
            new_vector[row_idx] += D.adj_data[i] * sm[idx];
        }
    }
    return new_vector;
}

std::vector<std::vector<float>>Filter(std::vector<float> rv, uint32_t k, spmv::io::CSCMatrix <float> &AD, uint32_t mx, uint32_t initial, uint32_t interval, uint32_t Ntot){
    std::vector<std::vector<float>> V(mx,std::vector<float>(Ntot));
    spmv::io::CSCMatrix<float> AD_diagnal;
    spmv::io::CSCMatrix<float> D;
    std::vector<float> AD_diagnal_val = {0.1};
    uint32_t sz;
    std::vector<std::vector<float>> sm_vec( mx , std::vector<float> (k));
    std::vector<float> sm_ot;
    std::vector<float> sm;
    std::vector<float> sm_norm;
    uint32_t count;

    sz = AD.num_rows;
    AD_diagnal = create_diagonal_matrix(AD_diagnal_val,sz);


    //V = zeros(mx, Ntot);

    //sm_vec = zeros(mx, k);

    //AD = AD .* 1.0

    //AD[diagind(AD, 0)] = AD[diagind(AD, 0)] .+ 0.1
    AD = sum(AD,AD_diagnal);

    //dg = sum(AD, dims = 1) .^ (-.5)

    //I2 = 1:sz

    //D = sparse(I2, I2, sparsevec(dg))
    D=SubRoutine001Filter(AD);

    //on = ones(Int, length(rv))

    //sm_ot = rv - ((dot(rv, on) / dot(on, on)) * on)
    sm_ot = SubRoutine002Filter(rv,AD.num_rows);

    //sm = sm_ot ./ norm(sm_ot);
    sm = SubRoutine003Filter(sm_ot,AD.num_rows);

    count = 1;

    spmv::io::CSRMatrix<float> AD_dash;
    spmv::io::CSRMatrix<float> D_dash;
	D_dash = csc2csr(D);
	AD_dash = csc2csr(AD);
	
    
	for(uint32_t loop=0;loop<k;loop++){
        //sm = D * sm
        sm = SubRoutine004Filter(D,sm);

        //sm = AD * sm
        sm = SubRoutine004Filter(AD,sm);

        //sm = D * sm
        sm = SubRoutine004Filter(D,sm);

        //sm_ot = sm - ((dot(sm, on) / dot(on, on)) * on)
        sm_ot = SubRoutine002Filter(sm,AD.num_rows);

        //sm_norm = sm_ot ./ norm(sm_ot);
        sm_norm = SubRoutine003Filter(sm_ot,AD.num_rows);
        for(uint32_t j=0;j<mx;j++){
            sm_vec[j][ loop] = sm_norm[j];
        }

    }
	

    //V = sm_vec[:, interval:interval:end]
    for(uint32_t j=0;j<mx;j++){
        for(uint32_t i=0;i<Ntot;i++){
            V[j][i]=sm_vec[j][(i*interval)+interval-1];
        }
    }

    return V;
}

void HyperEF(spmv::io::CSCMatrix <float> ar, uint32_t L, uint32_t R){
    spmv::io::CSCMatrix <float> ar_new;
    spmv::io::CSCMatrix <float> idx_mat;


    std::vector<float> Neff(ar.num_rows);
    std::vector<float> W(ar.num_cols);
    std::fill(Neff.begin(), Neff.end(), 0);
    std::fill(W.begin(), W.end(), 1.0);


    for(uint32_t loop=L;loop<=L;loop++){
        uint32_t mx;
        uint32_t initial;
        uint32_t SmS;
        uint32_t interval;
        uint32_t Nrv;
        uint32_t Nsm;
        uint32_t Ntot;
        std::vector <float> Qvec;
        std::vector<std::vector<float>> Eratio;

        mx = ar.num_rows;

        spmv::io::CSCMatrix <float> A;
        //star expansion
        A = StarW(ar,W);
        //computing the smoothed vectors
        initial = 0;

        SmS = 300;

        interval = 20;

        Nrv = 1;

        Nsm = int(float(SmS - initial) / float(interval));

        Ntot = Nrv * Nsm;

        //Qvec = zeros(Float64, 0);

        //Eratio = zeros(Float64, length(ar), Ntot)

        //SV = zeros(Float64, mx, Ntot)

        std::vector<std::vector<float>> SV(mx,std::vector<float>(Ntot));
        for(uint32_t ii=0;ii<Nrv;ii++){
            std::vector<std::vector<float>> sm;            //vector<vector<int>> vec( n , vector<int> (m, 0));
            //sm = zeros(mx, Nsm)
            std::vector<float> rv(A.num_rows);

            /*
            std::generate(rv.begin(), rv.end(), [&](){return ((float(get_random(0,INT32_MAX) / float(INT32_MAX)))- 0.5)*2.0;});
            */



            std::string line;
            std::ifstream myFile;            //creates stream myFile
            myFile.open("RVdata");  //opens .txt file
            uint32_t lines_read;
            lines_read=0;
            while (myFile.good()) {
                line.clear();
                getline(myFile, line);
                rv[lines_read++]=std::strtof(line.c_str(),nullptr);
                if(lines_read==A.num_rows)
                    break;
            }
            myFile.close();


            sm = Filter(rv, SmS, A, mx, initial, interval, Nsm);

            std::ofstream outfile ("Debug");
            for(uint32_t j=0;j<mx;j++){
                outfile << sm[j][Nsm-1] << std::endl;
            }
            outfile.close();

            //SV[:, (ii-1)*Nsm+1 : ii*Nsm] = sm
            for(uint32_t j=0;j<mx;j++){
                for(uint32_t i=0;i<Nsm;i++) {
                    SV[j][ii*Nsm+i] = sm[j][i];
                }
            }
        }
    }
}

#endif  // HYPEREF_H_