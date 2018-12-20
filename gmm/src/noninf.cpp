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

#include <math.h>
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
float compute_ELBO_svi(int j, MatrixXdr &mu_k, MatrixXdr &alpha, MatrixXdr &sigma_k, double sigma_q1, double sigma_q2, double sigma_e,double pi_q1, double pi_q2, MatrixXdr &XsXs, int miniB)
{
        MatrixXdr Xs =geno_matrix.block(0, j*miniB, g.Nsnp, miniB);
        MatrixXdr ys =pheno.block(j*miniB, 0, miniB, 1);

        float ll = -g.Nindv/2* log(sigma_e);
        MatrixXdr r = mu_k.cwiseProduct(alpha);
        MatrixXdr Xsr = ys - Xs.transpose()*r;
        MatrixXdr Xnorm = Xsr.cwiseProduct(Xsr);
        ll -= Xnorm.sum()*g.Nindv/2/sigma_e/miniB;
        for(int i =0; i<g.Nsnp ; i++)
        {
                float temp = alpha(i,0)*(sigma_k(i,0)+mu_k(i,0)*mu_k(i,0));
                temp = temp - alpha(i,0)*mu_k(i,0)*alpha(i,0)*mu_k(i,0);
                ll -= g.Nindv*XsXs(i,i)*temp /2 /sigma_e/miniB;
                if(alpha(i,0)!=0 && alpha(i,0)!=1 && i<g.Nsnp/2){
                float temp2 = alpha(i,0)*log(alpha(i,0)/pi_q1);
                ll -= temp2;

                float temp3 = (1-alpha(i,0))*log((1-alpha(i,0))/(1-pi_q1));
                ll -= temp3;
                }
		 if(alpha(i,0)!=0 && alpha(i,0)!=1 && i>=g.Nsnp/2){
                float temp2 = alpha(i,0)*log(alpha(i,0)/pi_q2);
                ll -= temp2;

                float temp3 = (1-alpha(i,0))*log((1-alpha(i,0))/(1-pi_q2));
                ll -= temp3;
                }
		float temp4; 
		if(i<g.Nsnp)
                	temp4 = 1+ log(sigma_k(i,0)/ sigma_q1/sigma_e) - (sigma_k(i,0)+ mu_k(i,0)*mu_k(i,0))/sigma_q1/sigma_e;
               else
			temp4 = 1+ log(sigma_k(i,0)/ sigma_q2/sigma_e) - (sigma_k(i,0)+ mu_k(i,0)*mu_k(i,0))/ sigma_q1/sigma_e; 
		 ll += alpha(i,0)*temp4 /2;
        }
        return ll;
}

void svi_step(MatrixXdr &mu_k, MatrixXdr &alpha, MatrixXdr &sigma_k, double &sigma_q1, double &sigma_q2, double &sigma_e, double &pi_q1, double &pi_q2, int miniB, int &t, float &ll, int step_a, int step_b, int iter)
{
		 for( int j=0; j<g.Nindv/miniB; j++)
                {
                        double step_temp = pow(t+step_a, step_b);
                        double step = 1/step_temp;
			//if step=1, force to forget 
			step=1; 
                        MatrixXdr Xs = geno_matrix.block(0, j*miniB, g.Nsnp, miniB);
                        MatrixXdr XsXs= Xs * Xs.transpose(); //M*M matrix	
			MatrixXdr ys = pheno.block(j*miniB, 0, miniB, 1);
                        MatrixXdr Xsys = Xs * ys;
     			double sigma_beta, pi;                    
			for( int i=0; i<g.Nsnp; i++)
                        {
				if(i<g.Nsnp/2){
					sigma_beta=sigma_q1;
					pi = pi_q1; 
				} 
				else 
				{
					sigma_beta=sigma_q2; 
					pi = pi_q2; 
				}
				double sigma_k_update = sigma_e/ (g.Nindv* XsXs(i,i)/miniB + 1/sigma_beta);
				sigma_k(i,0) =sigma_k_update;
				 MatrixXdr cur_snp = XsXs.block(0,i,g.Nsnp, 1);
	                        cur_snp = cur_snp.cwiseProduct(alpha);
        	                cur_snp = cur_snp.cwiseProduct(mu_k);
                	        MatrixXdr result = cur_snp.colwise().sum();
                        	result(0,0) = result(0,0) - cur_snp(i, 0);
                        	double mu_k_update = Xsys(i,0);
                        	mu_k_update = mu_k_update - result(0,0);
                        	mu_k_update = g.Nindv*sigma_k(i,0)* mu_k_update  / sigma_e/miniB;
	                        mu_k(i,0)= mu_k_update;
				//update alpha
				double temp = mu_k(i,0)*mu_k(i,0) / sigma_k(i,0)/2;
                        	double temp2 = sqrt(sigma_k(i,0))/sqrt(sigma_beta)/sqrt(sigma_e);
                        	double temp3 =  pi / (1-pi);
                        	double temp4 = exp(temp)*temp2 *temp3;
                        	double alpha_pre = alpha(i,0)/ (1-alpha(i,0));
                        	double update = pow(alpha_pre, 1-step);
                        	update *= pow(temp4, step);
                        	update *= pow(alpha_pre, 1-step);
                        	alpha(i,0) = temp4/(temp4+1);
				}
			//update theta	
	                MatrixXdr result = mu_k.cwiseProduct(mu_k) + sigma_k;
	                result  = result.cwiseProduct(alpha);
        	        MatrixXdr weight_sum = result.colwise().sum();
                	MatrixXdr alpha_sum = alpha.colwise().sum();
                	//sigma_beta =(1-step)*sigma_beta + step* weight_sum(0,0) /alpha_sum(0,0);
                	//pi = pi*(1-step) + step*alpha_sum(0,0) / g.Nsnp;
			sigma_q1=0; sigma_q2=0; pi_q1=0; pi_q2=0; 
			for(int i=0; i<g.Nsnp; i++)
			{
				if(i<g.Nsnp/2)
				{	
					sigma_q1 += result(i,0); 
					pi_q1 += alpha(i,0); 
				}
				else{
					sigma_q2 += result(i,0); 
					pi_q2 += alpha(i,0); 
				}
			}
			sigma_q1 = sigma_q1 / pi_q1/sigma_e; 
			sigma_q2 = sigma_q2 / pi_q2/sigma_e; 
			pi_q1 = pi_q1 / (g.Nsnp/2); 
			pi_q2 = pi_q2 / (g.Nsnp/2); 

              		MatrixXdr ysys = ys.transpose()*ys;
                	double sigma_e_update = ysys(0,0);
                	MatrixXdr weight_mu = alpha.cwiseProduct(mu_k);
                	MatrixXdr X_mu_alpha = Xs.transpose() * weight_mu;
                	MatrixXdr yX_mu_alpha = ys.transpose()* X_mu_alpha;
                	sigma_e_update  = sigma_e_update - yX_mu_alpha(0,0)*2;
                	MatrixXdr X_square =Xs.cwiseProduct(Xs);
                	MatrixXdr X_sigma_mu_alpha = X_square.transpose()*result;
                	MatrixXdr X_sigma_mu_alpha_sum = X_sigma_mu_alpha.colwise().sum();
                	sigma_e_update += X_sigma_mu_alpha_sum(0,0);
                	for(int b=0; b<miniB; b++)
                	{
                        	MatrixXdr ind_i = Xs.block(0, b, g.Nsnp, 1);
                        	ind_i = ind_i.cwiseProduct(mu_k);
                       		ind_i = ind_i.cwiseProduct(alpha);
                        	MatrixXdr temp = ind_i * ind_i.transpose();
                        	double temp_sum = temp.sum() - temp.trace();
                        	sigma_e_update += temp_sum;
                	}
			for(int m=0; m<g.Nsnp; m++)
			{
				double temp = alpha(m,0) * (mu_k(m,0)*mu_k(m,0) + sigma_k(m,0)); 
				if(m<g.Nsnp/2)
					temp = temp / sigma_q1; 
				else temp = temp/ sigma_q2; 
				sigma_e_update += temp; 
			}
			sigma_e_update = sigma_e_update / (g.Nindv + alpha.sum());
               		sigma_e = (1-step)*sigma_e + step * sigma_e_update;
			ll=compute_ELBO_svi(j, mu_k, alpha, sigma_k, sigma_q1, sigma_q2, sigma_e, pi_q1, pi_q2, XsXs, miniB); 
			cout<<iter<<"\t"<<t<< "\t"<<ll;
                	double vg = pi_q1 * g.Nsnp * sigma_q1 * sigma_e/2 + pi_q2 * g.Nsnp* sigma_q2 * sigma_e/2;
                	double h2g = vg / (vg + sigma_e);
               	 	cout<<"\t" <<pi_q1 <<"\t" << pi_q2 <<"\t"<<sigma_q1*sigma_e*pi_q1*g.Nsnp/2 << "\t"<< sigma_q2*sigma_e*pi_q2*g.Nsnp/2 << "\t" << vg <<"\t" <<h2g << endl;
			t++; 
		}	
		return; 		
}
void vem_step(MatrixXdr &mu_k, MatrixXdr &alpha, MatrixXdr &sigma_k, double &sigma_beta, double &sigma_e, double &pi, MatrixXdr &XX, MatrixXdr &Xy)
{
                for( int i=0; i<g.Nsnp; i++)
                {
                        sigma_k(i,0) = 1/ (XX(i,i)/sigma_e + 1/sigma_beta);
                        MatrixXdr cur_snp = XX.block(0, i, g.Nsnp, 1);
                        cur_snp = cur_snp.cwiseProduct(alpha);
                        cur_snp = cur_snp.cwiseProduct(mu_k);
                        MatrixXdr result = cur_snp.colwise().sum();
                        result(0,0) = result(0,0) - cur_snp(i, 0);
                        mu_k(i, 0) = Xy(i, 0);
                        mu_k(i, 0) = mu_k(i,0) - result(0,0);
                        mu_k(i,0) = sigma_k(i,0)* mu_k(i,0)  / sigma_e;
			//update alpha
		 	double temp = mu_k(i,0)*mu_k(i,0) / sigma_k(i,0)/2;
                        double temp2 = sqrt(sigma_k(i,0))/sqrt(sigma_beta);
                        double temp3 =  pi / (1-pi);
                        double temp4 = exp(temp)*temp2 *temp3;
                        alpha(i,0)  = temp4 / (temp4+1);
                }
                //update Theta
                MatrixXdr result = mu_k.cwiseProduct(mu_k) + sigma_k;
                result  = result.cwiseProduct(alpha);
                MatrixXdr weight_sum = result.colwise().sum();
                MatrixXdr alpha_sum = alpha.colwise().sum();
                sigma_beta = weight_sum(0,0) /alpha_sum(0,0);

                pi = alpha_sum(0,0) / g.Nsnp;
                cout<<"PI: "<<pi<<endl;

                MatrixXdr yy = pheno.transpose() * pheno;
                sigma_e = yy(0,0);
                MatrixXdr weight_mu = alpha.cwiseProduct(mu_k);
                MatrixXdr X_mu_alpha = geno_matrix.transpose() *  weight_mu;
                MatrixXdr yX_mu_alpha = pheno.transpose() * X_mu_alpha;
                sigma_e = sigma_e - yX_mu_alpha(0,0)*2;
                double elbo_temp=0;
                MatrixXdr X_square = geno_matrix.cwiseProduct(geno_matrix);
                MatrixXdr X_sigma_mu_alpha = X_square.transpose() * result;
                MatrixXdr X_sigma_mu_alpha_sum = X_sigma_mu_alpha.colwise().sum();
                sigma_e += X_sigma_mu_alpha_sum(0,0);
                elbo_temp  = X_sigma_mu_alpha_sum(0,0);
                //cout<<"sigma_e " <<sigma_e<<endl; 
		 for(int i=0; i<g.Nindv; i++)
                {
                        MatrixXdr ind_i = geno_matrix.block(0,i,g.Nsnp,1);
                        ind_i  = ind_i.cwiseProduct(mu_k);
                        ind_i = ind_i.cwiseProduct(alpha);
                        MatrixXdr temp = ind_i * ind_i.transpose();
                        double temp_sum = temp.sum()- temp.trace();
                        sigma_e += temp_sum;
                        elbo_temp += temp_sum;
                }
		 sigma_e = sigma_e / g.Nindv;
		return; 

}

float compute_ELBO_vem(MatrixXdr &mu_k, MatrixXdr &alpha, MatrixXdr &sigma_k, double sigma_beta, double sigma_e, double pi, MatrixXdr &XX)
{

	float ll = -g.Nindv/2* log(sigma_e); 
	MatrixXdr r = mu_k.cwiseProduct(alpha); 
	MatrixXdr Xr = pheno- geno_matrix.transpose()*r; 
	MatrixXdr Xrnorm = Xr.cwiseProduct(Xr); 
	ll -= Xrnorm.sum()/2/sigma_e; 
	for(int i =0; i<g.Nsnp ; i++)
	{
		float temp = alpha(i,0)*(sigma_k(i,0)+mu_k(i,0)*mu_k(i,0)); 
		temp = temp - alpha(i,0)*mu_k(i,0)*alpha(i,0)*mu_k(i,0); 
		ll -= XX(i,i)*temp /2 /sigma_e; 
		if(alpha(i,0)!=0 && alpha(i,0)!=1){
		float temp2 = alpha(i,0)*log(alpha(i,0)/pi); 
		ll -= temp2; 
		
		float temp3 = (1-alpha(i,0))*log((1-alpha(i,0))/(1-pi));
		ll -= temp3; 
		}
		float temp4 = 1+ log(sigma_k(i,0)/ sigma_beta) - (sigma_k(i,0)+ mu_k(i,0)*mu_k(i,0))/sigma_beta; 
		ll += alpha(i,0)*temp4 /2; 
	}	
	return ll; 
}
float compute_ELBO(MatrixXdr &mu_k, MatrixXdr &alpha, MatrixXdr &sigma_k, double sigma_beta, double sigma_e, double pi,double elbo_temp)
{
	MatrixXdr yy = pheno.transpose() * pheno;
	MatrixXdr mualpha = mu_k.cwiseProduct(alpha);  
	MatrixXdr Xmualpha = geno_matrix.transpose() * mualpha; 
	Xmualpha = pheno.cwiseProduct(Xmualpha); 
	float ll = - g.Nindv/2* std::log(sigma_e) -yy(0,0)/2/sigma_e + Xmualpha.sum()/2/sigma_e; 
	ll -= alpha.sum()* std::log(sigma_beta) / 2;  
	MatrixXdr mu_square = mu_k.cwiseProduct(mu_k); 
	MatrixXdr alpha_muS = alpha.cwiseProduct(mu_square + sigma_k); 
	ll -= alpha_muS.sum() / sigma_beta/2; 
	ll += std::log(pi)*alpha.sum(); 
	ll += std::log(1-pi)*(g.Nsnp - alpha.sum()); 
	ll -= alpha.sum()/2; 
	for(int i=0; i<g.Nsnp; i++)
 	{
		ll -= alpha(k,0)*std::log(sigma_k(k,0))/2; 
		ll -= alpha(k,0)*std::log(alpha(k,0)); 
		ll -= (1-alpha(k,0)) * std::log (1-alpha(k,0)); 		
	} 
	ll -= elbo_temp / 2 / sigma_e; 
	return ll; 

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
		g.generate_eigen_geno(geno_matrix, true); 
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
	MatrixXdr mu_k = MatrixXdr::Constant(g.Nsnp, 1, 0); 
	MatrixXdr sigma_k=MatrixXdr::Constant(g.Nsnp, 1, 1); 
	MatrixXdr alpha= MatrixXdr::Constant(g.Nsnp,1, 0.5); 

	MatrixXdr beta(g.Nsnp, 1); 
	double pi_q1=0.5;
	double pi_q2=0.5;  
	//exponential distribution
	boost::mt19937 gen; 
	boost::exponential_distribution<> dist(1); 
	boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > init(gen,dist);
	//two functions for now 
	double sigma_q1 = 1/ (init());
	double sigma_q2 = 1/ (init());  
	double sigma_e = 1/(init());
//loat l=init();
//	cout<<"Initializing prarameters..."<<endl<<"a: "<<a <<endl<<"l: "<<l <<endl; 
	
	float rho; 
	int iter=0; 

	float ll_det=1; 
	float prevll=0 ;
	MatrixXdr XX= geno_matrix*geno_matrix.transpose(); 
	MatrixXdr Xy = geno_matrix * pheno; 

	double step_a =1; 
	double step_b =1; 
	int t=0; 
	int miniB=g.Nindv/B; 
	cout<< "iter\tt\tELBO\tpi_q1\tpi_q2\tsigma_q1\tsigma_q2\tvg\th2g"<<endl; 
//	while(iter<=MAX_ITER ){ 
	while(iter<= MAX_ITER && ll_det > tol){
		

		/* one sample 
		MatrixXdr weight_mu = alpha.cwiseProduct(mu_k); 
		MatrixXdr Xsmualpha = Xs.transpose()*weight_mu; 
		double y_Xmualpha = pheno(j,0) - Xsmualpha(0,0); 
		double sigma_e_update = y_Xmualpha*y_Xmualpha; 
		
		MatrixXdr XsXs = Xs.cwiseProduct(Xs); 
		MatrixXdr varB = weight_mu.cwiseProduct(weight_mu); 
		MatrixXdr mumu = mu_k.cwiseProduct(mu_k)+sigma_k; 
		MatrixXdr varB_temp = alpha.cwiseProduct(mumu); 
		varB = varB_temp - varB; 		
		MatrixXdr varB_result = XsXs.cwiseProduct(varB); 
		sigma_e_update += varB_result(0,0); 
		sigma_e = (1-step)*sigma_e+ step*sigma_e_update; 

				*/

	/*	
		//variational EM step 
		vem_step(mu_k, alpha, sigma_k, sigma_beta, sigma_e, pi, XX, Xy); 
	//	variational EM ELBO
		float ll = compute_ELBO_vem(mu_k, alpha,sigma_k, sigma_beta, sigma_e, pi,XX); 
		cout<<iter<<"\t"<<iter<<"\t"<<ll<<"\t"<<pi ;
		double vg = sigma_beta * pi * g.Nsnp ; 
		double h2g  = vg / (vg+sigma_e); 
		cout<<"\t"<<vg<<"\t"<<h2g<<endl; 
	*/
		//svi step,ELBO
		float ll; 
		svi_step(mu_k, alpha, sigma_k, sigma_q1, sigma_q2, sigma_e, pi_q1, pi_q2, miniB,t, ll,step_a, step_b,iter); 
		if(iter>0)
			ll_det = ll - prevll;
		prevll = ll; 
		iter++; 


	}
	return 0;
}
