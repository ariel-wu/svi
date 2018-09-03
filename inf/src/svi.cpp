/** 
 Svi is written by Yue Ariel Wu
 UCLA 
*/
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector> 
#include <iomanip>

//#include <random>

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/QR>
#include "time.h"

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/random.hpp"
#include "boost/numeric/ublas/lu.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/triangular.hpp"

#include "genotype.h"
#include "mailman.h"
#include "arguments.h"
#include "helper.h"
#include "storage.h"

#if SSE_SUPPORT==1
	#define fastmultiply fastmultiply_sse
	#define fastmultiply_pre fastmultiply_pre_sse
#else
	#define fastmultiply fastmultiply_normal
	#define fastmultiply_pre fastmultiply_pre_normal
#endif
namespace ukblas =boost::numeric::ublas; 
using namespace Eigen;
using namespace std;

// Storing in RowMajor Form
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdr;

//Intermediate Variables
int blocksize;
double *partialsums;
double *sum_op;
double *yint_e;
double *yint_m;
double **y_e;
double **y_m;
//
//
struct timespec t0;

MatrixXdr covariate; 
MatrixXdr pheno;
genotype g;
MatrixXdr geno_matrix; //(p,n)
int MAX_ITER;
int k,p,n;
int k_orig;

MatrixXdr c;
MatrixXdr x; 
MatrixXdr v; 
MatrixXdr means; //(p,1)
MatrixXdr stds; //(p,1)
MatrixXdr sum2;
MatrixXdr sum;  


options command_line_opts;

bool debug = false;
bool check_accuracy = false;
bool var_normalize=true;
int accelerated_em=0;
double convergence_limit;
bool memory_efficient = false;
bool missing=false;
bool fast_mode = true;
bool text_version = false;
bool use_cov=false; 
bool reg = true;
bool gwas=false;  

vector<string> pheno_name; 

std::istream& newline(std::istream& in)
{
    if ((in >> std::ws).peek() != std::char_traits<char>::to_int_type('\n')) {
        in.setstate(std::ios_base::failbit);
    }
    return in.ignore();
}
int read_cov(bool std,int Nind, std::string filename, std::string covname){
	ifstream ifs(filename.c_str(), ios::in); 
	std::string line; 
	std::istringstream in; 
	int covIndex = 0; 
	std::getline(ifs,line); 
	in.str(line); 
	string b;
	vector<vector<int> > missing; 
	int covNum=0;  
	while(in>>b)
	{
		if(b!="FID" && b !="IID"){
		missing.push_back(vector<int>()); //push an empty row  
		if(b==covname && covname!="")
			covIndex=covNum; 
		covNum++; 
		}
	}
	vector<double> cov_sum(covNum, 0); 
	if(covname=="")
	{
		covariate.resize(Nind, covNum); 
		cout<< "Read in "<<covNum << " Covariates.. "<<endl;
	}
	else 
	{
		covariate.resize(Nind, 1); 
		cout<< "Read in covariate "<<covname<<endl;  
	}

	
	int j=0; 
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line);
		string temp;
		in>>temp; in>>temp; //FID IID 
		for(int k=0; k<covNum; k++){
			
			in>>temp;
			if(temp=="NA")
			{
				missing[k].push_back(j);
				continue;  
			} 
			double cur = atof(temp.c_str()); 
			if(cur==-9)
			{
				missing[k].push_back(j); 
				continue; 
			}
			if(covname=="")
			{
				cov_sum[k]= cov_sum[k]+ cur; 
				covariate(j,k) = cur; 
			}
			else
				if(k==covIndex)
				{
					covariate(j, 0) = cur;
					cov_sum[k] = cov_sum[k]+cur; 
				}
		} 
		j++;
	}
	//compute cov mean and impute 
	for (int a=0; a<covNum ; a++)
	{
		int missing_num = missing[a].size(); 
		cov_sum[a] = cov_sum[a] / (Nind - missing_num);

		for(int b=0; b<missing_num; b++)
		{
                        int index = missing[a][b];
                        if(covname=="")
                                covariate(index, a) = cov_sum[a];
                        else if (a==covIndex)
                                covariate(index, 0) = cov_sum[a];
                } 
	}
	if(std)
	{
		MatrixXdr cov_std;
		cov_std.resize(1,covNum);  
		MatrixXdr sum = covariate.colwise().sum();
		MatrixXdr sum2 = (covariate.cwiseProduct(covariate)).colwise().sum();
		MatrixXdr temp;
//		temp.resize(Nind, 1); 
//		for(int i=0; i<Nind; i++)
//			temp(i,0)=1;  
		for(int b=0; b<covNum; b++)
		{
			cov_std(0,b) = sum2(0,b) + Nind*cov_sum[b]*cov_sum[b]- 2*cov_sum[b]*sum(0,b);
			cov_std(0,b) =sqrt((Nind-1)/cov_std(0,b)) ;
			double scalar=cov_std(0,b); 
			for(int j=0; j<Nind; j++)
			{
				covariate(j,b) = covariate(j,b)-cov_sum[b];  
				covariate(j,b) =covariate(j,b)*scalar; 
			}
			//covariate.col(b) = covariate.col(b) -temp*cov_sum[b];
			
		}
	}	
	return covNum; 
}
int read_pheno2(int Nind, std::string filename,int pheno_idx){
//	pheno.resize(Nind,1); 
	ifstream ifs(filename.c_str(), ios::in); 
	
	std::string line;
	std::istringstream in;  
	int phenocount=0; 
	vector<vector<int> > missing; 
//read header
	std::getline(ifs,line); 
	in.str(line); 
	string b; 
	while(in>>b)
	{
		if(b!="FID" && b !="IID"){
			phenocount++;
			missing.push_back(vector<int>());  
			pheno_name.push_back(b); 
		}
	}
	if(pheno_idx !=0)
		pheno_name[0] = pheno_name[pheno_idx-1];   
	vector<double> pheno_sum(phenocount,0); 
	if(pheno_idx !=0)
		pheno.resize(Nind,1); 
	else
		pheno.resize(Nind, phenocount);
	int i=0;  
	while(std::getline(ifs, line)){
		in.clear(); 
		in.str(line); 
		string temp;
		//fid,iid
		//todo: fid iid mapping; 
		//todo: handle missing phenotype
		in>>temp; in>>temp; 
		for(int j=0; j<phenocount;j++) {
			in>>temp;
			if(temp=="NA")
			{
				missing[j].push_back(i); 
				continue; 
			} 
			double cur= atof(temp.c_str()); 
			if(pheno_idx==0){
				pheno(i,j)=cur; 
				pheno_sum[j] = pheno_sum[j]+cur; 
			}
			else if(j == (pheno_idx-1))
			{
				pheno(i,0)=cur; 
				pheno_sum[j]= pheno_sum[j]+cur; 
			}
		}
		i++;
	}
	for(int a=0; a<phenocount; a++)
	{
		int missing_num= missing[a].size(); 
		double	pheno_avg = pheno_sum[a]/(Nind- missing_num); 
		//cout<<"pheno "<<a<<" avg: "<<pheno_avg<<endl; 
		for(int b=0 ; b<missing_num; b++)
		{
			int index = missing[a][b]; 
			pheno(index, a)= pheno_avg; 
		}
		if(pheno_idx!=0)
			return 1;  
	}
	return phenocount; 
}


void multiply_y_pre_fast(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means){
	
	for(int k_iter=0;k_iter<Ncol_op;k_iter++){
		sum_op[k_iter]=op.col(k_iter).sum();		
	}

			//cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Starting mailman on premultiply"<<endl;
			cout << "Nops = " << Ncol_op << "\t" <<g.Nsegments_hori << endl;
			cout << "Segment size = " << g.segment_size_hori << endl;
			cout << "Matrix size = " <<g.segment_size_hori<<"\t" <<g.Nindv << endl;
			cout << "op = " <<  op.rows () << "\t" << op.cols () << endl;
		}
	#endif


	//TODO: Memory Effecient SSE FastMultipy

	for(int seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
		mailman::fastmultiply(g.segment_size_hori,g.Nindv,Ncol_op,g.p[seg_iter],op,yint_m,partialsums,y_m);
		int p_base = seg_iter*g.segment_size_hori; 
		for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++ ){
			for(int k_iter=0;k_iter<Ncol_op;k_iter++) 
				res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
		}
	}

	int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply(last_seg_size,g.Nindv,Ncol_op,g.p[g.Nsegments_hori-1],op,yint_m,partialsums,y_m);		
	int p_base = (g.Nsegments_hori-1)*g.segment_size_hori;
	for(int p_iter=p_base; (p_iter<p_base+g.segment_size_hori) && (p_iter<g.Nsnp) ; p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++) 
			res(p_iter,k_iter) = y_m[p_iter-p_base][k_iter];
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Ending mailman on premultiply"<<endl;
		}
	#endif


	if(!subtract_means)
		return;

	for(int p_iter=0;p_iter<p;p_iter++){
 		for(int k_iter=0;k_iter<Ncol_op;k_iter++){		 
			res(p_iter,k_iter) = res(p_iter,k_iter) - (g.get_col_mean(p_iter)*sum_op[k_iter]);
			if(var_normalize)
				res(p_iter,k_iter) = res(p_iter,k_iter)/(g.get_col_std(p_iter));		
 		}		
 	}	

}

void multiply_y_post_fast(MatrixXdr &op_orig, int Nrows_op, MatrixXdr &res,bool subtract_means){

	MatrixXdr op;
	op = op_orig.transpose();

	if(var_normalize && subtract_means){
		for(int p_iter=0;p_iter<p;p_iter++){
			for(int k_iter=0;k_iter<Nrows_op;k_iter++)		
				op(p_iter,k_iter) = op(p_iter,k_iter) / (g.get_col_std(p_iter));		
		}		
	}

	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Starting mailman on postmultiply"<<endl;
		}
	#endif
	
	int Ncol_op = Nrows_op;

	//cout << "ncol_op = " << Ncol_op << endl;

	int seg_iter;
	for(seg_iter=0;seg_iter<g.Nsegments_hori-1;seg_iter++){
		mailman::fastmultiply_pre(g.segment_size_hori,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);
	}
	int last_seg_size = (g.Nsnp%g.segment_size_hori !=0 ) ? g.Nsnp%g.segment_size_hori : g.segment_size_hori;
	mailman::fastmultiply_pre(last_seg_size,g.Nindv,Ncol_op, seg_iter * g.segment_size_hori, g.p[seg_iter],op,yint_e,partialsums,y_e);

	for(int n_iter=0; n_iter<n; n_iter++)  {
		for(int k_iter=0;k_iter<Ncol_op;k_iter++) {
			res(k_iter,n_iter) = y_e[n_iter][k_iter];
			y_e[n_iter][k_iter] = 0;
		}
	}
	
	#if DEBUG==1
		if(debug){
			print_time (); 
			cout <<"Ending mailman on postmultiply"<<endl;
		}
	#endif


	if(!subtract_means)
		return;

	double *sums_elements = new double[Ncol_op];
 	memset (sums_elements, 0, Nrows_op * sizeof(int));

 	for(int k_iter=0;k_iter<Ncol_op;k_iter++){		
 		double sum_to_calc=0.0;		
 		for(int p_iter=0;p_iter<p;p_iter++)		
 			sum_to_calc += g.get_col_mean(p_iter)*op(p_iter,k_iter);		
 		sums_elements[k_iter] = sum_to_calc;		
 	}		
 	for(int k_iter=0;k_iter<Ncol_op;k_iter++){		
 		for(int n_iter=0;n_iter<n;n_iter++)		
 			res(k_iter,n_iter) = res(k_iter,n_iter) - sums_elements[k_iter];		
 	}


}

void multiply_y_pre_naive_mem(MatrixXdr &op, int Ncol_op ,MatrixXdr &res){
	for(int p_iter=0;p_iter<p;p_iter++){
		for(int k_iter=0;k_iter<Ncol_op;k_iter++){
			double temp=0;
			for(int n_iter=0;n_iter<n;n_iter++)
				temp+= g.get_geno(p_iter,n_iter,var_normalize)*op(n_iter,k_iter);
			res(p_iter,k_iter)=temp;
		}
	}
}

void multiply_y_post_naive_mem(MatrixXdr &op, int Nrows_op ,MatrixXdr &res){
	for(int n_iter=0;n_iter<n;n_iter++){
		for(int k_iter=0;k_iter<Nrows_op;k_iter++){
			double temp=0;
			for(int p_iter=0;p_iter<p;p_iter++)
				temp+= op(k_iter,p_iter)*(g.get_geno(p_iter,n_iter,var_normalize));
			res(k_iter,n_iter)=temp;
		}
	}
}

void multiply_y_pre_naive(MatrixXdr &op, int Ncol_op ,MatrixXdr &res){
	res = geno_matrix * op;
}

void multiply_y_post_naive(MatrixXdr &op, int Nrows_op ,MatrixXdr &res){
	res = op * geno_matrix;
}

void multiply_y_post(MatrixXdr &op, int Nrows_op ,MatrixXdr &res,bool subtract_means){
    if(fast_mode)
        multiply_y_post_fast(op,Nrows_op,res,subtract_means);
    else{
		if(memory_efficient)
			multiply_y_post_naive_mem(op,Nrows_op,res);
		else
			multiply_y_post_naive(op,Nrows_op,res);
	}
}

void multiply_y_pre(MatrixXdr &op, int Ncol_op ,MatrixXdr &res,bool subtract_means){
    if(fast_mode)
        multiply_y_pre_fast(op,Ncol_op,res,subtract_means);
    else{
		if(memory_efficient)
			multiply_y_pre_naive_mem(op,Ncol_op,res);
		else
			multiply_y_pre_naive(op,Ncol_op,res);
	}
}



int main(int argc, char const *argv[]){


	pair<double,double> prev_error = make_pair(0.0,0.0);
	double prevnll=0.0;

	parse_args(argc,argv);

	
	//TODO: Memory effecient Version of Mailman

	memory_efficient = command_line_opts.memory_efficient;
	text_version = command_line_opts.text_version;
	fast_mode = command_line_opts.fast_mode;
	missing = command_line_opts.missing;
	reg = command_line_opts.reg;
	gwas=command_line_opts.gwas; 
	if(gwas)
		fast_mode=false; //for now, gwas need the genotype matrix, and compute kinship constructed with one chrom leave out 
	if(!reg)
		fast_mode=false; //force save whole genome if non randomized  
	if(text_version){
		if(fast_mode)
			g.read_txt_mailman(command_line_opts.GENOTYPE_FILE_PATH,missing);
		else
			g.read_txt_naive(command_line_opts.GENOTYPE_FILE_PATH,missing);
	}
	else{
		g.read_plink(command_line_opts.GENOTYPE_FILE_PATH,missing,fast_mode);
		
	}

	//TODO: Implement these codes.
	if(missing && !fast_mode){
		cout<<"Missing version works only with mailman i.e. fast mode\n EXITING..."<<endl;
		exit(-1);
	}
	if(fast_mode && memory_efficient){
		cout<<"Memory effecient version for mailman EM not yet implemented"<<endl;
		cout<<"Ignoring Memory effecient Flag"<<endl;
	}
	if(missing && var_normalize){
		cout<<"Missing version works only without variance normalization\n EXITING..."<<endl;
		exit(-1);
	}

	MAX_ITER=100; 
    //	MAX_ITER =  command_line_opts.max_iterations ; 
	int B = command_line_opts.batchNum; 
	int pheno_idx = command_line_opts.pheno_idx; 
	k_orig = command_line_opts.num_of_evec ;
	debug = command_line_opts.debugmode ;
	float tr2= command_line_opts.tr2; 
	check_accuracy = command_line_opts.getaccuracy;
	var_normalize = true; 
	accelerated_em = command_line_opts.accelerated_em;
	k = k_orig + command_line_opts.l;
	k = (int)ceil(k/10.0)*10;
	command_line_opts.l = k - k_orig;
	p = g.Nsnp;
	n = g.Nindv;
	bool toStop=false;
		toStop=true;
	srand((unsigned int) time(0));
	
	means.resize(p,1);
	stds.resize(p,1);
	sum2.resize(p,1); 
	sum.resize(p,1); 

//	geno_matrix.resize(p,n); 
//	g.generate_eigen_geno(geno_matrix, var_normalize); 

	if(!fast_mode && !memory_efficient){
		geno_matrix.resize(p,n);
//		g.generate_eigen_geno(geno_matrix,var_normalize);
		g.generate_eigen_geno(geno_matrix, false); 
		cout<<geno_matrix.data()<<endl; 
		cout<<geno_matrix.rows(); 
		cout<<geno_matrix.cols(); 

	}
	
	
	for(int i=0;i<p;i++){
		means(i,0) = g.get_col_mean(i);
		stds(i,0) =1/ g.get_col_std(i);
		sum2(i,0) =g.get_col_sum2(i); 
		sum(i,0)= g.get_col_sum(i); 
	}



//	cout<<"printing means: "<<endl<<means<<endl; 
//	cout<<"printing std: "<<endl<<stds<<endl; 	
	ofstream c_file;

	cout<<"Running on Dataset of "<<g.Nsnp<<" SNPs and "<<g.Nindv<<" Individuals"<<endl;

	#if SSE_SUPPORT==1
		if(fast_mode)
			cout<<"Using Optimized SSE FastMultiply"<<endl;
	#endif


	//get geno
	//cout<<g.get_geno(0,0,false);
	//read phenotype
	//
	//
	std::string filename=command_line_opts.PHENOTYPE_FILE_PATH; 
	int pheno_num= read_pheno2(g.Nindv, filename, pheno_idx);
	int cov_num=0 ;
	if(filename=="")
	{	
		cout<<"No Phenotype File Specified"<<endl;
		return 0 ; 
	}
	cout<< "Read in "<<pheno_num << " phenotypes"<<endl; 
	if(pheno_idx!=0)
		cout<<"Using phenotype "<<pheno_name[pheno_idx-1]<<endl; 
	

	MatrixXdr y_sum=pheno.colwise().sum(); 
	MatrixXdr y_mean = y_sum/g.Nindv;
	for(int i=0; i<g.Nindv; i++) 
		pheno.block(i,0,1,pheno_num) =pheno.block(i,0,1,pheno_num) - y_mean; //center phenotype	
	y_sum=pheno.colwise().sum();
	

//model
	float hyper[4];
	for(int i=0; i<4; i++)
		hyper[i]=1e-3; 
	float tol = 1e-6; 
	MatrixXdr beta = MatrixXdr::Constant(g.Nsnp, 1, 0); 
	MatrixXdr betaSigma = MatrixXdr::Constant(g.Nsnp, g.Nsnp, 1); 

	//exponential distribution
	boost::mt19937 gen; 
	boost::exponential_distribution<> dist(1); 
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > init(gen,dist);

	float a=init();
	float l=init();
	a=2.575836; l=0.1306393;  
	cout<<"Initializing prarameters..."<<endl<<"a: "<<a <<endl<<"l: "<<l <<endl; 
	
	float rho; 
	int iter=0; 

	float ll_det=1; 
	float prevll=0 ;
//	while(iter <= MAX_ITER){
	while(iter<= MAX_ITER && ll_det > tol){
		rho = a/l; 
		MatrixXdr temp = MatrixXdr::Identity(g.Nsnp,g.Nsnp); 
		MatrixXdr betaSigmaInv = geno_matrix * geno_matrix.transpose() + rho* temp; 
		betaSigma =betaSigmaInv.inverse();

		beta = geno_matrix* pheno; 
		beta = betaSigma * beta; 
		betaSigma = betaSigma / l; 

		MatrixXdr A = betaSigma * geno_matrix; 
		MatrixXdr B = geno_matrix.transpose()*A; 
		
		MatrixXdr temp_vec = geno_matrix.transpose() * beta; 
		temp_vec = pheno - temp_vec; 
		double temp_l = temp_vec.squaredNorm(); //TODO: check with R code if doing desired operation
		l = 0.5 * (temp_l + B.trace() )+ hyper[3]; 
		l = l/ (0.5 * g.Nindv + hyper[2] -1); 
		l = 1/l; 

		double temp_a = beta.squaredNorm(); 
		a=0.5*(betaSigma.trace()+ temp_a) + hyper[1]; 
		a=a/(0.5*g.Nsnp + hyper[0]-1); 
		a = 1/a; 
		

		//compute ELBO
		
		float ll = -(0.5 * g.Nindv + hyper[2]-1) + (0.5 *g.Nindv + hyper[2]-1)*std::log(l) - (0.5* g.Nsnp + hyper[0]-1) + (0.5*g.Nsnp + hyper[0]-1)*std::log(a) + 0.5*std::log(betaSigma.trace()); 
		if(iter>0)
			ll_det = ll - prevll;
		prevll = ll; 
		cout<<iter<< ": ELBO: "<<ll<<"   ";  
		
		float s2g = g.Nsnp/a; 
		float s2e = 1/l; 
		float h2g = s2g/ (s2g + s2e); 
//		cout<<"V(G): "<<s2g<<endl;
//	        cout<<"V(E): "<<s2e<<endl;
        	cout<<"h2g: "<<h2g<<endl;
		iter++; 


}
	return 0;
}
