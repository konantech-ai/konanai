/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "local_library.h"
#include "../session/session.h"
#include "../components/component.h"
#include "../components/kmodel.h"
#include "../utils/kutil.h"
#include "../utils/kutil.h"

#include "assert.h"

int KaiLocalLibrary::ms_checkCodeSave = 12716160;


//hs.cho
#ifndef KAI2021_WINDOWS
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif

/*
_InitModelInfo KaiLocalLibrary::ms_initModuleNames[] = {
	{ "abalone", true, "/MLP", "kaggle_abalone", "csv_loader", "mlp_6_12", 0, "gsd" },
	{ "pulsar", false, "/MLP", "kaggle_star", "csv_loader", "mlp_6_12", 0, "gsd"  },
	{ "flower", true, "/MLP", "kaggle_flowers", "file_loader", "mlp_16_32_64", 0, "adam" },
	{ "office31", false, "/MLP", "ms_office31", "file_loader", "mlp_16_32_64", "dual_classify", "adam" },
	{ "flower_cnn", true, "/CNN", "kaggle_flowers", "file_loader", "cnn_3step", 0, "adam" },
	{ "office31_cnn", false, "/CNN", "ms_office31", "file_loader", "cnn_3step", "dual_classify", "adam" },
	{ "flower_inception", false, "/CNN/Advanced", "kaggle_flowers", "file_loader", "inception_v3", 0, "adam" },
	{ "flower_resnet", false, "/CNN/Advanced", "kaggle_flowers", "file_loader", "residual_34", 0, "adam" },
	{ "bert", true, "/Commercial", "korean_news", "bert_loader", "bert_large", "bert_loss", "adam" },
	{ "yolo", false, "/Commercial", "coco", "yolo_loader", "yolo_3", "yolo_loss", "adam" },
};
*/

_InitModelInfo KaiLocalLibrary::ms_initModuleNames[] = {
	{ "abalone", true, "/MLP", "kaggle_abalone", "shuffle_loader", "mlp_6_12", 0, "gsd_3" },
	{ "pulsar", false, "/MLP", "kaggle_pulsar", "shuffle_loader", "mlp_6_12", 0, "gsd_3"  },
	{ "flower", false, "/MLP", "kaggle_flower_100", "flower_loader", "mlp_1024_128_16", 0, "adam_4" },
	{ "flower_cnn", true, "/CNN", "kaggle_flower_96", "flower_loader", "cnn_3step", "flower_loss", "adam_3" },
};

_InitComponentInfo KaiLocalLibrary::ms_iniComponentInfo[] = {
	{ "abalone", "{'loss':'regression'}" },
	{ "kaggle_abalone", "{'source_url':'https://www.kaggle.com/rodolfomendes/abalone-dataset', 'filename': 'abalone.csv', \
                          'reader':'csv_reader', 'csv_header':true, 'onehot_column':'[[0,3]]', 'input_shape':'[4177,10]', 'output_shape':[4177,1]}" },
	{ "shuffle_loader", "{'shuffle':true, 'train_ratio':'0.8', 'validate_ratio':'0.1', 'test_ratio':'0.1'}" },
	{ "mlp_6_12", "'net':'[[full', width:6}], [full, width:12]]'" },
	{ "gsd_3", "{'algorithm':'gsd', 'learning_rate':'0.003'}" },

	{ "pulsar", "{'loss':'binary'}" },
	{ "kaggle_pulsar", "{'source_url':'https://www.kaggle.com/colearninglounge/predicting-pulsar-starintermediate', 'filename': 'pulsar_data_train.csv', \
                         'reader':'csv_reader', 'csv_header':true, 'input_shape':'[12528,8]', 'output_shape':'[12528,1]', 'data_dist':'[769,1055,784,734,984]'}" },

	{ "flower", "{'loss':'classify', 'targets':'[daisy, dandelion, rose, sunflower, tulip]'}" },
	{ "kaggle_flower_100", "{'source_url':'https://www.kaggle.com/colearninglounge/predicting-pulsar-starintermediate', \
                         'reader':'folder_reader', 'file_format':'image', 'file_resolution':'various', 'input_shape':'[4326,100,100,3]', 'output_shape':'[4326,5]', 'data_dist':'[769,1055,784,734,984]'}" },
	{ "flower_loader", "{'shuffle_per_class':true, 'train_ratio':'0.7', 'validate_ratio':'0.1', 'test_ratio':'0.2'}" },
	{ "mlp_1024_128_16", "'net':'[[full, width:1024], [full, width:128], [full, width:16]]'" },
	{ "adam_4", "{'algorithm':'adam', 'learning_rate':'0.004', 'epsilon':'0.99', 'ro1':'0.1', 'ro2':'0.01'}" },

	{ "flower_cnn", "{'loss':'custom', 'targets':'[daisy, dandelion, rose, sunflower, tulip]'}" },
	{ "kaggle_flower_96", "{'source_url':'https://www.kaggle.com/colearninglounge/predicting-pulsar-starintermediate', \
                         'reader':'folder_reader', 'file_format':'image', 'file_resolution':'various', 'input_shape':'[4326,96,96,3]', 'output_shape':'[4326,1]', 'data_dist':'[769,1055,784,734,984]'}" },
	{ "flower_loader", "{'shuffle_per_class':true, 'train_ratio':'0.7', 'validate_ratio':'0.1', 'test_ratio':'0.2'}" },
	{ "cnn_3step", "'net':'[[cnn, kernel:[3,3], chn:6], [max, stride:2], [cnn, kernel:[3,3], chn:12], [max, stride:2], [cnn, kernel:[3,3], chn:24], [global_average]]'" },
	{ "flower_loss", "{'loss', 'softmax_cross_entropy_idx(est,ans)'}" },
	{ "adam_3", "{'algorithm':'adam', 'learning_rate':'0.003', 'epsilon':'0.99', 'ro1':'0.1', 'ro2':'0.01'}" },
};

std::map<KString, KString> KaiLocalLibrary::ms_initComponentInfoMap;

KaiLocalLibrary::KaiLocalLibrary(KaiSession* pSession, KString sLibName, KString sPassword) : KaiLibrary(pSession, sLibName) {
	m_pRootFolder = NULL;

	m_load_library_file();

	if (m_sPassword != sPassword) throw KaiException(KERR_LOCAL_LIB_PASSWORD_MISMATCH);

	m_pCurrFolder = m_pRootFolder;

	m_bTouched = false;
	m_bDestroyed = false;

	m_checkCode = ms_checkCodeLocal;
}

KaiLocalLibrary::KaiLocalLibrary(KaiSession* pSession, KString sLibName, KString sPassword, Ken_inst_mode enumInstallMode, KStrList slModelNames) : KaiLibrary(pSession, sLibName) {
	if (!ms_isValidLibName(sLibName)) throw KaiException(KERR_BAD_NAME_USED, "LocalLibrary", sLibName);
	if (!ms_isValidPassword(sPassword)) throw KaiException(KERR_BAD_PASSWORD_FORMAT, "LocalLibrary");

	m_sPassword = sPassword;
	m_sVersion = pSession->GetVersion();

	m_pRootFolder = NULL;

	m_create_library_file(enumInstallMode, slModelNames);

	m_bTouched = false;
	m_bDestroyed = false;

	m_checkCode = ms_checkCodeLocal;
}

KaiLocalLibrary::~KaiLocalLibrary() {
	m_save_library_file(false);
	delete m_pRootFolder;
	m_checkCode = 0;
}

KaiLocalLibrary* KaiLocalLibrary::HandleToPointer(KHObject hObject, KaiSession* pSession) {
	if (hObject == NULL) throw KaiException(KERR_NULL_HANDLE_USED, "LocalLibrary");
	KaiLocalLibrary* pLocalLib = (KaiLocalLibrary*)hObject;
	if (pLocalLib->m_checkCode != ms_checkCodeLocal) throw KaiException(KERR_INVALIDL_HANDLE_USED, "LocalLibrary");
	if (pLocalLib->m_pOwnSession != pSession) throw KaiException(KERR_BAD_SESSION_USED, "LocalLibrary");
	return pLocalLib;
}

int KaiLocalLibrary::issueComponentId() {
	return m_nNextComponentId++;
}

void KaiLocalLibrary::GetInstallModels(KStrList* pslModuleNames) {
	int nInitModelCnt = sizeof(ms_initModuleNames) / sizeof(ms_initModuleNames[0]);
	for (int n = 0; n < nInitModelCnt; n++) {
		pslModuleNames->push_back(ms_initModuleNames[n].modelName);
	}
}

void KaiLocalLibrary::setName(KString sNewLibName) {
	if (m_sLibName == sNewLibName) throw KaiException(KERR_RENAME_TO_OLD_NAME, "LocalLibrary", sNewLibName);
	if (!ms_isValidLibName(sNewLibName)) throw KaiException(KERR_BAD_NAME_USED, "LocalLibrary", sNewLibName);
	
	KString sOldLibname = m_sLibName;
	m_sLibName = sNewLibName;
	m_save_library_file(true);
	m_remove_old_library_file(sOldLibname);
}

void KaiLocalLibrary::changePassword(KString sOldPassword, KString sNewPassword) {
	if (sOldPassword != m_sPassword) throw KaiException(KERR_LOCAL_LIB_PASSWORD_MISMATCH);

	m_sPassword = sNewPassword;
	m_save_library_file(true);
}

void KaiLocalLibrary::installModels(KStrList slModels) {
	m_checkModelNames(slModels);
	m_installModels(slModels);
}

void KaiLocalLibrary::save() {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiLocalLibrary::destory() {
	m_bDestroyed = true;
	m_pOwnSession->DeleteLocalLibary(m_sLibName, true);
}

KString KaiLocalLibrary::GetVersion() {
	return m_sVersion;
}

void KaiLocalLibrary::close(bool bSave) {
	assert(0); // save 여부를어떻게 지원할지 정해 처리할 것
	if (!m_pOwnSession->CloseLocalLibary(this)) throw KaiException(KERR_INTERNAL_ERROR_IN_CLOSING_LOCLIB, m_sLibName);
}


KPathString KaiLocalLibrary::getCurrPath() {
	return m_pCurrFolder->getFolderPath();
}

void KaiLocalLibrary::setCurrPath(KPathString sCurrPath) {
	m_pCurrFolder = m_seekFolder(sCurrPath);
}

void KaiLocalLibrary::createFolder(KPathString sNewPath, bool bThrowOnExist) {
	KString sFolderToCreate;
	LocalFolder* pParent = m_seekFolder(sNewPath, &sFolderToCreate);
	pParent->createSubFolder(sFolderToCreate, bThrowOnExist);
}

void KaiLocalLibrary::renameFolder(KPathString sFolderPath, KString sNewName) {
	LocalFolder* pFolderToRename = m_seekFolder(sFolderPath);

	if (pFolderToRename->getParent()->hasNamedChild(pFolderToRename->getName())) throw KaiException(KERR_DUPLICATED_SUBFOLDER_NAME, sFolderPath, sNewName);

	pFolderToRename->rename(sNewName);
}

void KaiLocalLibrary::moveFolder(KPathString sOldPath, KPathString sDestPath) {
	LocalFolder* pFolderToMove = m_seekFolder(sOldPath);
	LocalFolder* pDstParent = m_seekFolder(sDestPath);

	if (pDstParent->hasNamedChild(pFolderToMove->getName())) throw KaiException(KERR_DUPLICATED_SUBFOLDER_NAME, sOldPath, pFolderToMove->getName());
	if (pFolderToMove->isAncestorOf(pDstParent)) throw KaiException(KERR_CANNOT_MOVE_TO_DESCENDANT, sOldPath, sDestPath);

	pFolderToMove->moveFolder(pDstParent);
}

void KaiLocalLibrary::deleteFolder(KPathString sFolderPath) {
	LocalFolder* pFolderToDel = m_seekFolder(sFolderPath);

	if (pFolderToDel == m_pRootFolder) throw KaiException(KERR_CANNOT_DELETE_ROOT_FOLDER);
	if (pFolderToDel == m_pCurrFolder) m_pCurrFolder = m_pCurrFolder->getParent();

	pFolderToDel->destroy();
}

void KaiLocalLibrary::list(KJsonStrList* pjlComponents, KPathString sPath, bool recursive) {
	pjlComponents->clear();
	LocalFolder* pFolder = m_seekFolder(sPath);
	pFolder->listAllComponents(pjlComponents, recursive);
}

void KaiLocalLibrary::listSubFolders(KPathStrList* pslSubFolders, KPathString sPath, bool recursive) {
	pslSubFolders->clear();
	LocalFolder* pFolder = m_seekFolder(sPath);
	pFolder->listFolders(pslSubFolders, recursive);
}

/*
KaiComponent* KaiLocalLibrary::downloadComponent(Ken_component_type componentType, KPathString sComponentPath) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
	KString sComponentName;
	LocalFolder* pFolder = m_seekFolder(sComponentPath, &sComponentName);
	KaiComponent* pComponent = pFolder->getNamedComponent(sComponentName, componentType);
	return pComponent;
}
*/

void KaiLocalLibrary::registComponent(Ken_component_type componentType, KPathString sNewComponentPath, KaiComponent* pComponent) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
	/*
	pComponent->typeCheck(componentType);
	KString sNewComponentName;
	LocalFolder* pFolder = m_seekFolder(sNewComponentPath, &sNewComponentName);
	pFolder->registComponent(sNewComponentName, pComponent);
	*/
}

void KaiLocalLibrary::updateComponent(Ken_component_type componentType, KPathString sComponentPath, KaiComponent* pComponent) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
	/*
	KaiComponent* pCurrComponent = m_seekComponent(sComponentPath, componentType);
	pCurrComponent->update(pComponent);
	*/
}

void KaiLocalLibrary::listComponents(Ken_component_type componentType, KJsonStrList* pslComponentInfo, KPathString sPath, bool recursive) {
	pslComponentInfo->clear();
	LocalFolder* pFolder = m_seekFolder(sPath);
	pFolder->listComponents(pslComponentInfo, recursive, componentType);
}

void KaiLocalLibrary::setProperty(KPathString sComponentlPath, KPathString sProperty, KString sValue) {
	throw KaiException(KERR_UNIMPEMENTED_YET);
}

void KaiLocalLibrary::moveComponent(KPathString sCurComponentPath, KPathString sDestFolder) {
	KaiComponentInfo* pComponent = seekComponent(sCurComponentPath);
	LocalFolder* pDestFolder = m_seekFolder(sDestFolder);
	pComponent->moveFolder(pDestFolder);
}

void KaiLocalLibrary::renameComponent(KPathString sCurComponentPath, KString sNewName) {
	KaiComponentInfo* pComponent = seekComponent(sCurComponentPath);
	pComponent->rename(sNewName);
}

void KaiLocalLibrary::deleteComponent(KPathString sComponentlPath) {
	KaiComponentInfo* pComponent = seekComponent(sComponentlPath);
	pComponent->destory();
}

LocalFolder* KaiLocalLibrary::m_seekOrCreateFolder(KPathString sPath) {
	bool bIsAbs;
	KStrList pieces = m_splitPathPieces(sPath, NULL, &bIsAbs);
	LocalFolder* pFolder = bIsAbs ? m_pRootFolder : m_pCurrFolder;
	for (auto it=pieces.begin(); it!=pieces.end(); it++) {
		KString sPiece = *it;
		if (sPiece == "..") pFolder = pFolder->getParent();
		else if (sPiece == ".."); // do nothing
		else pFolder = pFolder->createSubFolder(sPiece, false);
	}

	return pFolder;
}

LocalFolder* KaiLocalLibrary::m_seekFolder(KPathString sPath, KString* pLastPiece) {
	bool bAbs;
	KStrList slPieces = m_splitPathPieces(sPath, pLastPiece, &bAbs);
	return (bAbs ? m_pRootFolder : m_pCurrFolder)->seekFolder(slPieces);
}

KaiComponentInfo* KaiLocalLibrary::seekComponent(KPathString sPath) { // , Ken_component_type componentType) {
	KString sComponentName;
	LocalFolder* pFolder = m_seekFolder(sPath, &sComponentName);
	return pFolder->getNamedComponent(sComponentName); // , componentType);
}

KaiComponentInfo* KaiLocalLibrary::createComponent(KPathString sPath, Ken_component_type componentType) {
	KString sComponentName;
	LocalFolder* pFolder = m_seekFolder(sPath, &sComponentName);
	KaiComponentInfo* pComponentInfo = pFolder->addComponent(componentType, sComponentName, true);
	return pComponentInfo;
}

KStrList KaiLocalLibrary::m_splitPathPieces(KPathString sFolderPath, KString* pLastPiece, bool* pbIsAbsolutePath) {
	KStrList result;
	KString lastPiece;

	if (sFolderPath == "") {
		if (pLastPiece) throw KaiException(KERR_BAD_EMPTY_STRING_FOR_PATH);
		*pbIsAbsolutePath = false;
		return result;
	}
#ifdef KAI2021_WINDOWS
	std::replace(sFolderPath.begin(), sFolderPath.end(), '\\', '/');
#endif 

	const char* str = sFolderPath.c_str();
	
	if (pbIsAbsolutePath) *pbIsAbsolutePath = str[0] == '/';

	do {
		const char* begin = str;

		while (*str != '/' && *str) str++;

		if (lastPiece != "") result.push_back(lastPiece);
		lastPiece = KString(begin, str);
	} while (0 != *str++);

	if (lastPiece != "") {
		if (pLastPiece) *pLastPiece = lastPiece;
		else result.push_back(lastPiece);
	}

	return result;
}

void KaiLocalLibrary::m_create_library_file(Ken_inst_mode inst_mode, KStrList slModelNames) {
	KString sFilePath = m_pOwnSession->GetLocalLibFilePath(m_sLibName);

	m_nNextComponentId = 1;

	m_pRootFolder = new LocalFolder(this, NULL, "");
	m_pCurrFolder = m_pRootFolder;

	int nInitModelCnt = sizeof(ms_initModuleNames) / sizeof(ms_initModuleNames[0]);

	switch (inst_mode) {
	case Ken_inst_mode::none:
		break;
	case Ken_inst_mode::standard:
		for (int n = 0; n < nInitModelCnt; n++) {
			if (ms_initModuleNames[n].isStandatd) {
				m_installModel(n);
			}
		}
		break;
	case Ken_inst_mode::full:
		for (int n = 0; n < nInitModelCnt; n++) {
			m_installModel(n);
		}
		break;
	case Ken_inst_mode::custom:
		m_checkModelNames(slModelNames);
		m_installModels(slModelNames);
		break;
	}

	m_save_library_file(true);
}

void KaiLocalLibrary::m_installModel(int nth) {
	if (ms_initComponentInfoMap.empty()) {
		int nInfoCount = sizeof(ms_iniComponentInfo) / sizeof(ms_iniComponentInfo[0]);

		for (int n = 0; n < nInfoCount; n++) {
			KString name = ms_iniComponentInfo[n].componentName;
			KString props = ms_iniComponentInfo[n].componentProps;
			ms_initComponentInfoMap[name] = props;
		}
	}

	_InitModelInfo& pInfo = ms_initModuleNames[nth];

	LocalFolder* pFolder = m_seekOrCreateFolder(pInfo.modelPath);

	KaiComponentInfo* pModelInfo = pFolder->addComponent(Ken_component_type::model, pInfo.modelName, false);
	
	if (pInfo.dataset) {
		KaiComponentInfo* pDatasetInfo = pFolder->addComponent(Ken_component_type::dataset, pInfo.dataset, false);
		pDatasetInfo->setInitialProps(ms_initComponentInfoMap[pInfo.dataset]);
		pModelInfo->setProperty("dataset", pDatasetInfo->getComponentId());
	}

	if (pInfo.dataloader) {
		KaiComponentInfo* pDataloaderInfo = pFolder->addComponent(Ken_component_type::dataloader, pInfo.dataloader, false);
		pDataloaderInfo->setInitialProps(ms_initComponentInfoMap[pInfo.dataloader]);
		pModelInfo->setProperty("dataloader", pDataloaderInfo->getComponentId());
	}

	if (pInfo.network) {
		KaiComponentInfo* pNetworkInfo = pFolder->addComponent(Ken_component_type::network, pInfo.network, false);
		pNetworkInfo->setInitialProps(ms_initComponentInfoMap[pInfo.network]);
		pModelInfo->setProperty("network", pNetworkInfo->getComponentId());
	}

	if (pInfo.expression) {
		KaiComponentInfo* pExpressionInfo = pFolder->addComponent(Ken_component_type::expression, pInfo.expression, false);
		pExpressionInfo->setInitialProps(ms_initComponentInfoMap[pInfo.expression]);
		pModelInfo->setProperty("expression", pExpressionInfo->getComponentId());
	}

	if (pInfo.optimizer) {
		KaiComponentInfo* pOptimizerInfo = pFolder->addComponent(Ken_component_type::optimizer, pInfo.optimizer, false);
		pOptimizerInfo->setInitialProps(ms_initComponentInfoMap[pInfo.optimizer]);
		pModelInfo->setProperty("optimizer", pOptimizerInfo->getComponentId());
	}
}

void KaiLocalLibrary::m_checkModelNames(KStrList slModels) {
	int nInitModelCnt = sizeof(ms_initModuleNames) / sizeof(ms_initModuleNames[0]);

	for (auto it = slModels.begin(); it != slModels.end(); it++) {
		bool bFound = false;
		for (int n = 0; n < nInitModelCnt; n++) {
			if (ms_initModuleNames[n].modelName == (KString) *it) {
				bFound = true;
				break;
			}
		}
		if (!bFound) throw KaiException(KERR_UNKNOWN_INSTALL_MODEL_NAME, *it);
	}
}

void KaiLocalLibrary::m_installModels(KStrList slModels) {
	int nInitModelCnt = sizeof(ms_initModuleNames) / sizeof(ms_initModuleNames[0]);

	for (auto it = slModels.begin(); it != slModels.end(); it++) {
		for (int n = 0; n < nInitModelCnt; n++) {
			if (ms_initModuleNames[n].modelName == (KString) *it) {
				m_installModel(n);
				break;
			}
		}
	}
}

bool KaiLocalLibrary::ms_isValidLibName(KString sLibName) {
	size_t length = sLibName.size();

	for (size_t n = 0; n < length; n++) {
		int ch = sLibName[n];
		if (!isalnum(ch) && ch != '_' && ch != '-') return false;
	}

	return true;
}

bool KaiLocalLibrary::ms_isValidPassword(KString sPassword) {
	if (strchr(sPassword.c_str(), '\n')) return false;
	if (sPassword.size() > 80) return false;
	return true;
}

void KaiLocalLibrary::m_remove_old_library_file(KString sOldLibName) {
	KString sFilePath = m_pOwnSession->GetLocalLibFilePath(sOldLibName);
	kutil.remove_all(sFilePath);
}

void KaiLocalLibrary::m_load_library_file() {
	KString sFilePath = m_pOwnSession->GetLocalLibFilePath(m_sLibName);

	FILE* fid = NULL;

	if (fopen_s(&fid, sFilePath.c_str(), "rb") != 0) throw KaiException(KERR_FAIL_TO_OPEN_LOCAL_LIB_FILE, sFilePath);

	int nCheckCode = (int) kutil.read_int(fid);

	if (nCheckCode != ms_checkCodeSave) throw KaiException(KERR_BROKEN_LOCAL_LIB_FILE, sFilePath);

	m_sPassword = kutil.read_str(fid);
	m_sVersion = kutil.read_str(fid);

	m_nNextComponentId = (int) kutil.read_int(fid);

	KBindTable bindTab;
	KComponentMap componentMap;

	m_pRootFolder = new LocalFolder(this, NULL, "");
	m_pRootFolder->unserialize(fid); // (fid, bindTab, componentMap);

	fclose(fid);

	/*
	for (auto it = bindTab.begin(); it != bindTab.end(); it++) {
		_BindInfo& info = *it;
		KaiComponentInfo* pSubjectComponent = componentMap[info.subject];
		KaiComponentInfo* pObjectComponent = componentMap[info.object];
		int relation = info.relation;

		pSubjectComponent->bindRelation(pObjectComponent, relation);
	}
	*/

	m_bTouched = false;
}

void KaiLocalLibrary::m_save_library_file(bool bMandatory) {
	if (!bMandatory && (!m_bTouched || m_bDestroyed)) return;

	KString sFilePath = m_pOwnSession->GetLocalLibFilePath(m_sLibName);

	FILE* fid = NULL;

	if (fopen_s(&fid, sFilePath.c_str(), "wb") != 0) throw KaiException(KERR_FAIL_TO_SAVE_LOCAL_LIB_FILE, sFilePath);

	kutil.save_int(fid, ms_checkCodeSave);

	kutil.save_str(fid, m_sPassword);
	kutil.save_str(fid, m_sVersion);

	kutil.save_int(fid, m_nNextComponentId);

	m_pRootFolder->serialize(fid);

	fclose(fid);

	m_bTouched = false;
}
