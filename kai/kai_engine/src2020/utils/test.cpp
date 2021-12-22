/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#define KAI_CUDA_EXPORTS
#ifdef KAI2021_WINDOWS //hs.cho
#include "test.h"
#include "../core/log.h"

int get_phone_num() { return 7569; }

ANSWERCB cb_answer;
ALLOCCB cb_alloc;
FREECB cb_free;

//int params[2];

void ms_produce(void* aux) {
    int* params = (int*)aux;
    int n = params[0];
    int m = params[1];
    
    cb_free(params);

    for (int k = 0; k < 10; k++) {
        //(rand() % 2 == 0) ? n++ : m++;
        (k % 2 == 0) ? n++ : m++;
        cb_answer(n, m);
    }

}

int SetCallback(ANSWERCB fp, ALLOCCB fa, FREECB ff, int n, int m) {
    cb_answer = fp;
    cb_alloc= fa;
    cb_free = ff;

    if (cb_answer)
    {
        cb_answer(3752, 7569);
        int* params = cb_alloc(2);
        params[0] = n; 
        params[1] = m;
        ms_produce(params);
        //new std::thread(ms_produce, params); // params);
        return cb_answer(n, m);
    }
    return 0;
}

// For X DevAPI for C++
//#include <mysqlx/xdevapi.h>

// For X DevAPI for C
#include <mysqlx/xapi.h>

// For legacy API
//#include "stdio.h"
//#include "winsock2.h"
//#include "mysql/jdbc.h"

//#pragma comment(lib, "libmysql.lib")

/*
sql::SQLString hostName = "127.0.0.1"; // jdbc:mysql://www.2woo.net/ewoo";
//sql::SQLString hostName = "tcpip:mysql://211.239.121.216/ewoo";
sql::SQLString userName = "ewoo";
sql::SQLString password = "ejqnfdj#$13";
*/

/*
sql::SQLString hostName = "localhost";
sql::SQLString userName = "root";
sql::SQLString password = "tail99#headMYS";
*/

MysqlConn::MysqlConn() {
}

MysqlConn::~MysqlConn() {
}

void MysqlConn::test() {
    try {
        /*
        char* host = "211.239.121.216";
        char* user = "ewoo";
        char* password = "ejqnfdj#$13";
        char* database = "";
        int port = 3306;
        */

        char* host = "127.0.0.1";
        char* user = "root";
        //char* password = "tail99#headMYS";
        char* password = "kal#2021";
        char* database = "";
        int port = 33060;

        mysqlx_error_t* error = NULL;

        mysqlx_session_t* sess = mysqlx_get_session(host, port, user, password, database, &error);

        if (error) {
            logger.Print("%d: %s", mysqlx_error_num(error), mysqlx_error_message(error));
            mysqlx_free(error);
            return;
        }

        char* query = "create database kal";
            
        mysqlx_result_t* result = mysqlx_sql(sess, query, MYSQLX_NULL_TERMINATED);
        
        if (result) {
            mysqlx_result_free(result);
        }
        else {
            error = mysqlx_error(sess);
            logger.Print("CREATE DB FAILURE: %d: %s", mysqlx_error_num(error), mysqlx_error_message(error));
            mysqlx_free(error);
        }

        int check1 = mysqlx_session_valid(sess);
        logger.Print("check1 = %d", check1);

        mysqlx_session_close(sess);
    }
    catch (...) {
       std::cerr << "Error Connecting to MySQL Platform";
       //throw KaiException(KERR_ASSERT);
    }
}

/*
void MysqlConn::test() {
    try
    {
        sql::Driver* driver = sql::mysql::get_driver_instance();

        //sql::SQLString url("tcpip:mysql://www.2woo.net/ewoo/");

        sql::Connection* conn = driver->connect(hostName, userName, password);

        if (conn) {
            logger.Print("connected");
            conn->close();
        }
        else {
            logger.Print("dpdpng");
        }
    }
    catch (sql::SQLException& e)
    {
        std::cerr << "Error Connecting to MySQL Platform: " << e.what() << std::endl;
        //throw KaiException(KERR_ASSERT);
    }
}
*/

/*
void MysqlConn::test() {
    MYSQL* connection = NULL;
    MYSQL conn;
    MYSQL_RES* sql_result;
    MYSQL_ROW sql_row;

    if (mysql_init(&conn) == NULL)
    {
        logger.Print("mysql_init() error!");
    }

    connection = mysql_real_connect(&conn, host, user, pw, db, 3306, (const char*)NULL, 0);
    if (connection == NULL)    // 연결 에러 처리
    {
        logger.Print("%d 에러 : %s, %d", mysql_errno(&conn), mysql_error(&conn));
        return 1;
    }
    else
    {
        logger.Print("연결 성공");    // 연결 성공 메시지 출력

        if (mysql_select_db(&conn, db))    // 데이터베이스 선택
        {
            logger.Print("%d 에러 : %s, %d", mysql_errno(&conn), mysql_error(&conn));
            return 1;
        }

        char* query = "select *from korea";
        int state = 0;

        state = mysql_query(connection, query);
        if (state == 0)
        {
            sql_result = mysql_store_result(connection);            // Result Set에 저장
            while ((sql_row = mysql_fetch_row(sql_result)) != NULL)    // Result Set에서 1개씩 배열에 가져옴
            {
                logger.Print("%s %s %s %s", sql_row[0], sql_row[1], sql_row[2], sql_row[3]);    // 저장된 배열을 출력
            }
            mysql_free_result(sql_result);        // Result Set 해제한다
        }

        mysql_close(connection);        // db서버 종료
    }

    return 0;
}
*/
#endif
