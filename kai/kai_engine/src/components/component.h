/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "component_info.h"
#include "../library/object.h"

class KaiLibrary;
class KaiExecContext;

class KaiComponent : public KaiObject {
public:
	KaiComponent(KaiSession* pSession, Ken_component_type ctype, Ken_object_type otype);
	KaiComponent(KaiSession* pSession, Ken_component_type ctype, Ken_object_type otype, KaiDict kwArgs);
	KaiComponent(KaiSession* pSession, Ken_component_type ctype, Ken_object_type otype, KaiLibrary* pLib, KaiComponentInfo* pComponentInfo);

	virtual ~KaiComponent();

	static KaiComponent* HandleToPointer(KHObject hObject);
	static KaiComponent* HandleToPointer(KHObject hObject, KaiSession* pSession);

	void close();

	KaiSession* get_session() { return m_pSession; }

	int get_ref_count() { return m_nRefCount; }
	//KaiComponent* add_ref_count() { m_nRefCount++; return this; }

	KBool isDirty() { return m_bDirty; }
	void refreshDirty();

	virtual void regist(KaiLibrary* pLib, KString sNewComponentlPath);
	virtual void touch();
	virtual void update();
	
	// depreciated: bExternal argumnent
	virtual void bind(KaiComponent* pComponent, KString relation, bool bExternal, bool bUnique);
	virtual void unbind(KaiComponent* pComponent, bool bReport = true);

	virtual void set_property(KString sKey, KaiValue kValue);

	KaiValue get_property(KString sKey);
	KaiValue get_property(KString sKey, KaiValue def);

	static KaiValue get_info_prop(KaiDict info, KString sKey, KaiValue def);
	static KaiValue get_set_info_prop(KaiDict info, KString sKey, KaiValue def);

	KString get_desc();

	void set_property_dict(KaiDict kwArgs);

	void dump_property(KString sKey, KString sTitle);

	void destroy_without_report();

	const KaiDict& getPropDictRef() { return m_propDict; }

	virtual KaiDict resolve_copy_properties(KaiDict call_info);

	void collect_properties(KaiComponent* pInst, KString sCompName);

	//void push_properties(KaiDict& dict);

	virtual void set_datafeed_callback(Ken_datafeed_cb_event cb_event, void* pCbInst, void* pCbFunc, void* pCbAux);
	virtual void set_train_callback(Ken_train_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux);
	virtual void set_visualize_callback(Ken_visualize_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux);
	virtual void set_test_callback(Ken_test_cb_event cb_event, void* pCbInst, void* pCbReport, void* pCbAux);
	//virtual void set_predict_callback(void* pCbInst, void* pCbFuncStart, void* pCbFuncData, void* pCbFuncEnd, void* pCbAux);

protected:
	KaiDict m_copy_properties();

	void m_set_default(KString sKey, KaiValue kDefault); // 생성자 호출 인자 통해 깂이 지정되지 않은 경우에 한해 디폴트 값 설정

	KaiValue m_seek_property(KString sKey, KaiValue def, KaiDict kwArgs1, KaiDict kwArgs2=KaiDict());

	void m_pushCallbackPropert(KaiComponent* pDest, KaiDict cbFamilies);
	void m_regist_cb_info(KString sFamilyKey, KString sEventKey, KaiList cb_info);
	void m_save_cb_info(Ken_cb_family cb_family, int cb_event, void* pCbInst, void* pCbFunc, void* pCbAux);

protected:
	KaiSession* m_pSession;
	KaiLibrary* m_pOwnLibrary;	// maybe NULL (download가 아닌 create로 생성된 경우)

	int m_nComponentId;				// maybe 0 (download가 아닌 create로 생성된 경우)
	int m_nComponentSeqId;			// 무조건 0 아닌 일련번호

	static int ms_nNextSeqID;

	Ken_component_type m_componentType;

	bool m_bDirty;

	KaiDict m_propDict;

	int m_checkCode;
	int m_checkCodeComponent;

	static int ms_checkCodeComponent;
};
